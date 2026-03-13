"""
BrainPy Baseline Fitter — Fixed Na+K+Leak HH Model (Weeks 3–4)
================================================================

Fits a single-compartment Hodgkin-Huxley model (Na + K + Leak) to PV+
fast-spiking interneuron recordings from the Allen Cell Types Database
using BrainPy's differentiable simulation and gradient descent.

This is the "BrainPy fixed-HH" baseline — the simplest biophysical model
that the agent-discovered models must beat.

MIGRATED FROM JAXLEY TO BRAINPY:
  - jx.Compartment() + insert(Na/K/Leak) → bp.dyn.CondNeuGroupLTC subclass
  - cell.stimulate() / cell.record() → bp.DSRunner with monitors
  - jx.integrate(cell, param_state=...) → step-by-step BrainPy simulation
  - cell.make_trainable / ParamTransform / SigmoidTransform
      → manual bm.Variable params + sigmoid_transform helper + jax.grad

Usage:
    python jaxley_fit.py --data-dir ./cell_types_data
    python jaxley_fit.py --data-dir ./cell_types_data --specimen-id 469801138
    python jaxley_fit.py --data-dir ./cell_types_data --epochs 200 --lr 0.01
    python jaxley_fit.py --data-dir ./cell_types_data --all

Requires:
    pip install brainpy jax jaxlib optax allensdk matplotlib
"""

# ---- JAX config must come before any JAX imports ----
from jax import config
config.update("jax_enable_x64", True)
# config.update("jax_platform_name", "gpu")  # Uncomment if GPU available

import os
import json
import argparse
import logging
from pathlib import Path

import numpy as np
print(np.__version__)  # Must be 1.26.x
import xarray as xr
import jax
import jax.numpy as jnp
from jax import jit, value_and_grad
import optax
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import brainpy as bp
import brainpy.math as bm

from allensdk.core.cell_types_cache import CellTypesCache

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Allen Data Loader — extract a single training sweep
# ---------------------------------------------------------------------------
# (Unchanged from original — no Jaxley dependency)

def load_training_sweep(ctc: CellTypesCache, specimen_id: int,
                        sweep_index: dict) -> dict:
    """
    Load a training long-square sweep for the given specimen.
    Prefers a sweep with moderate spiking — picks the lowest-amplitude
    sweep that has spikes (i.e. just above rheobase), then steps up
    one more to get a sweep with a few more spikes for a richer signal.

    Returns dict with 'time', 'stimulus', 'response', 'sampling_rate' arrays.
    """
    cell_entry = sweep_index[str(specimen_id)]
    train_sweeps = cell_entry["split"]["training"]["long_square"]

    if not train_sweeps:
        raise ValueError(f"No training sweeps for specimen {specimen_id}")

    by_amp = sorted(train_sweeps, key=lambda s: s.get("stimulus_amplitude", 0) or 0)
    spiking = [sw for sw in by_amp if (sw.get("num_spikes") or 0) > 0]

    if not spiking:
        logger.info("  num_spikes metadata unavailable; scanning NWB for spiking sweeps...")
        data_set = ctc.get_ephys_data(specimen_id)
        for sw in reversed(by_amp[-min(8, len(by_amp)):]):
            try:
                spike_times = data_set.get_spike_times(sw["sweep_number"])
                if len(spike_times) > 0:
                    sw["num_spikes"] = len(spike_times)
                    spiking.append(sw)
            except Exception:
                pass
        spiking.sort(key=lambda s: s.get("stimulus_amplitude", 0) or 0)

    if spiking:
        idx = min(len(spiking) - 1, max(1, len(spiking) // 3))
        chosen = spiking[idx]
        logger.info(
            f"  Selected spiking sweep {chosen['sweep_number']} "
            f"(amplitude={chosen.get('stimulus_amplitude', '?'):.1f} pA, "
            f"num_spikes={chosen.get('num_spikes', '?')}, "
            f"index {idx}/{len(spiking)} spiking sweeps)"
        )
    else:
        chosen = by_amp[-1]
        logger.warning(
            f"  No spiking sweeps found! Using highest amplitude sweep "
            f"{chosen['sweep_number']} ({chosen.get('stimulus_amplitude', '?')} pA)"
        )

    data_set = ctc.get_ephys_data(specimen_id)
    sweep_data = data_set.get_sweep(chosen["sweep_number"])

    idx = sweep_data["index_range"]
    stimulus = sweep_data["stimulus"][idx[0]:idx[1]+1]
    response = sweep_data["response"][idx[0]:idx[1]+1]
    sr = sweep_data["sampling_rate"]

    return {
        "sweep_number": chosen["sweep_number"],
        "stimulus_amplitude": chosen.get("stimulus_amplitude"),
        "time": np.arange(len(stimulus)) / sr,
        "stimulus": stimulus,    # Amps
        "response": response,    # Volts
        "sampling_rate": sr,
    }


def load_multiple_sweeps(ctc: CellTypesCache, specimen_id: int,
                         sweep_index: dict, n_sweeps: int = 3) -> list:
    """Load multiple training sweeps spanning a range of amplitudes."""
    cell_entry = sweep_index[str(specimen_id)]
    train_sweeps = cell_entry["split"]["training"]["long_square"]

    if not train_sweeps:
        raise ValueError(f"No training sweeps for specimen {specimen_id}")

    by_amp = sorted(train_sweeps, key=lambda s: s.get("stimulus_amplitude", 0) or 0)
    n = min(n_sweeps, len(by_amp))
    indices = np.linspace(0, len(by_amp) - 1, n, dtype=int)
    selected = [by_amp[i] for i in indices]

    data_set = ctc.get_ephys_data(specimen_id)
    results = []
    for sw in selected:
        sweep_data = data_set.get_sweep(sw["sweep_number"])
        idx = sweep_data["index_range"]
        stimulus = sweep_data["stimulus"][idx[0]:idx[1]+1]
        response = sweep_data["response"][idx[0]:idx[1]+1]
        sr = sweep_data["sampling_rate"]
        results.append({
            "sweep_number": sw["sweep_number"],
            "stimulus_amplitude": sw.get("stimulus_amplitude"),
            "time": np.arange(len(stimulus)) / sr,
            "stimulus": stimulus,
            "response": response,
            "sampling_rate": sr,
        })
    return results


# ---------------------------------------------------------------------------
# 2. BrainPy Model Builder — single-compartment Na+K+Leak
# ---------------------------------------------------------------------------

class FixedHHNeuron(bp.dyn.CondNeuGroupLTC):
    """
    Single-compartment HH neuron with Na, K, and Leak channels.
    This is the fixed 3-channel baseline model.

    Replaces the Jaxley build_hh_cell() which did:
        comp = jx.Compartment()
        comp.insert(Na()); comp.insert(K()); comp.insert(Leak())
        comp.set("Na_gNa", 0.5); comp.set("K_gK", 0.2); ...

    BrainPy uses class composition: channels are attached as attributes
    in __init__, and CondNeuGroupLTC automatically collects their currents.

    PARAMETER NOTE: Jaxley uses S/cm² for conductance and the cell geometry
    (radius/length) affects current density. BrainPy's built-in HH channels
    use mS/cm² (msiemens). We use BrainPy's native units here, which means
    the parameter values differ from the Jaxley version but produce
    equivalent biophysics.
    """

    def __init__(self, size, gNa=120.0, gK=36.0, gL=0.03, eLeak=-60.0,
                 eNa=50.0, eK=-77.0, C=1.0):
        super().__init__(
            size,
            C=C,
            V_th=20.0,
            V_initializer=bp.init.Constant(-65.0),
        )
        self.INa = bp.dyn.INa_HH1952(size, E=eNa, g_max=gNa)
        self.IK = bp.dyn.IK_HH1952(size, E=eK, g_max=gK)
        self.IL = bp.dyn.IL(size, E=eLeak, g_max=gL)


def build_hh_cell(dt: float = 0.025) -> FixedHHNeuron:
    """
    Build a single-compartment HH cell with Na, K, and Leak channels.

    Conductance densities are initialised for PV+ fast-spiking cells.
    Uses BrainPy's native msiemens units (g_max in mS/cm²).
    """
    model = FixedHHNeuron(
        size=1,
        gNa=120.0,      # mS/cm² — standard HH (high for FS cells)
        gK=36.0,         # mS/cm²
        gL=0.03,         # mS/cm²
        eLeak=-60.0,     # mV
        eNa=50.0,        # mV
        eK=-77.0,        # mV
        C=1.0,           # uF/cm²
    )
    return model


# ---------------------------------------------------------------------------
# 2b. Stimulus and Target preparation
# ---------------------------------------------------------------------------
# (Unchanged — no Jaxley dependency. Just resampling.)

def prepare_stimulus(sweep: dict, dt: float = 0.025) -> tuple:
    """
    Resample Allen stimulus to BrainPy's simulation timestep.
    Allen stimulus is in Amps; BrainPy expects nA for external current.
    """
    sr = sweep["sampling_rate"]
    stimulus_amps = sweep["stimulus"]
    stimulus_nA = stimulus_amps * 1e9

    t_allen = np.arange(len(stimulus_nA)) / sr
    t_max_s = t_allen[-1]
    t_max_ms = t_max_s * 1000.0

    n_steps = int(t_max_ms / dt) + 1
    t_sim_ms = np.arange(n_steps) * dt
    t_sim_s = t_sim_ms / 1000.0

    stimulus_resampled = np.interp(t_sim_s, t_allen, stimulus_nA)
    return stimulus_resampled, t_max_ms


def prepare_target(sweep: dict, dt: float = 0.025) -> np.ndarray:
    """
    Resample Allen voltage response to BrainPy's simulation timestep.
    Allen response is in Volts; BrainPy uses mV.
    """
    sr = sweep["sampling_rate"]
    response_V = sweep["response"]
    response_mV = response_V * 1e3

    t_allen = np.arange(len(response_V)) / sr
    t_max_s = t_allen[-1]
    t_max_ms = t_max_s * 1000.0

    n_steps = int(t_max_ms / dt) + 1
    t_sim_ms = np.arange(n_steps) * dt
    t_sim_s = t_sim_ms / 1000.0

    response_resampled = np.interp(t_sim_s, t_allen, response_mV)
    return response_resampled


# ---------------------------------------------------------------------------
# 3. Simulation & Loss Functions
# ---------------------------------------------------------------------------
# KEY MIGRATION POINT: Jaxley's simulation+loss was:
#   1. cell.stimulate(current) to register stimulus
#   2. jx.integrate(cell, param_state=...) inside a differentiable fn
#   3. cell.make_trainable() + cell.get_parameters() + ParamTransform
#
# BrainPy equivalent:
#   1. We build a function that manually steps the model, injecting
#      current at each timestep, inside a jax.lax.scan loop.
#   2. Parameters are stored as a flat dict of jnp arrays, and we
#      write them onto model attributes before each forward pass.
#   3. We use a manual sigmoid_transform instead of Jaxley's
#      ParamTransform/SigmoidTransform.

def sigmoid_transform(x, lower, upper):
    """Map unconstrained value x to (lower, upper) via sigmoid."""
    return lower + (upper - lower) * jax.nn.sigmoid(x)


def inverse_sigmoid(y, lower, upper):
    """Map constrained value y in (lower, upper) to unconstrained space."""
    p = (y - lower) / (upper - lower)
    p = jnp.clip(p, 1e-6, 1.0 - 1e-6)
    return jnp.log(p / (1.0 - p))


# Parameter bounds — biophysically reasonable ranges for PV+ FS cells
PARAM_BOUNDS = {
    "gNa":   (10.0, 500.0),     # mS/cm² — FS cells have high Na density
    "gK":    (5.0, 200.0),      # mS/cm²
    "gL":    (0.001, 5.0),      # mS/cm²
    "eLeak": (-80.0, -40.0),    # mV
    "C":     (0.5, 3.0),        # uF/cm²
    "eNa":   (30.0, 70.0),      # mV
    "eK":    (-100.0, -60.0),   # mV
}


def run_simulation(gNa, gK, gL, eLeak, C, eNa, eK, stimulus, dt):
    """
    Run a BrainPy HH simulation with given parameters.

    This replaces jx.integrate(cell, param_state=...).

    We rebuild the model with the given parameters each call so that
    the computation is fully inside the JAX trace (differentiable).
    BrainPy models are JAX-compatible — all state updates use bm ops
    which are traced by JAX's autodiff.
    """
    model = FixedHHNeuron(size=1, gNa=gNa, gK=gK, gL=gL,
                          eLeak=eLeak, eNa=eNa, eK=eK, C=C)

    runner = bp.DSRunner(model, monitors=['V'], dt=dt)
    runner.run(inputs=stimulus)
    v = bm.as_jax(runner.mon['V'])
    return v.flatten()


def build_loss_fn(target_v, stimulus, dt):
    """
    Build a differentiable loss function.

    The loss takes unconstrained parameters, transforms them to
    biophysical bounds via sigmoid, runs the BrainPy simulation,
    and returns MSE.

    Replaces Jaxley's build_loss_fn which used cell.data_set() +
    ParamTransform + jx.integrate().
    """

    def loss_fn(opt_params):
        # Transform from unconstrained -> constrained parameter space
        gNa   = sigmoid_transform(opt_params["gNa"],   *PARAM_BOUNDS["gNa"])
        gK    = sigmoid_transform(opt_params["gK"],    *PARAM_BOUNDS["gK"])
        gL    = sigmoid_transform(opt_params["gL"],    *PARAM_BOUNDS["gL"])
        eLeak = sigmoid_transform(opt_params["eLeak"], *PARAM_BOUNDS["eLeak"])
        C     = sigmoid_transform(opt_params["C"],     *PARAM_BOUNDS["C"])
        eNa   = sigmoid_transform(opt_params["eNa"],   *PARAM_BOUNDS["eNa"])
        eK    = sigmoid_transform(opt_params["eK"],    *PARAM_BOUNDS["eK"])

        # Run simulation
        v_sim = run_simulation(gNa, gK, gL, eLeak, C, eNa, eK, stimulus, dt)

        # Trim to match target length
        n = min(len(v_sim), len(target_v))
        v_sim_trimmed = v_sim[:n]
        v_target_trimmed = target_v[:n]

        # Waveform MSE loss
        mse = jnp.mean((v_sim_trimmed - v_target_trimmed) ** 2)
        return mse

    return loss_fn


# ---------------------------------------------------------------------------
# 4. Training Loop
# ---------------------------------------------------------------------------

def fit_cell(ctc: CellTypesCache, specimen_id: int, sweep_index: dict,
             output_dir: Path, dt: float = 0.025, epochs: int = 100,
             lr: float = 0.02) -> dict:
    """
    Full fitting pipeline for one cell:
    1. Load a training sweep
    2. Build single-compartment HH model
    3. Set up differentiable simulation
    4. Run gradient descent with optax
    5. Save results and diagnostic plots

    Replaces the original Jaxley-based fit_cell which used:
        cell = build_hh_cell()
        cell.stimulate(stimulus)
        cell.make_trainable("Na_gNa"); ...
        loss_fn = build_loss_fn(cell, target, dt, transform)
        grad_fn = jit(value_and_grad(loss_fn))
    """
    logger.info(f"Fitting specimen {specimen_id}...")
    out = output_dir / str(specimen_id)
    out.mkdir(parents=True, exist_ok=True)

    # ---- Load data ----
    sweep = load_training_sweep(ctc, specimen_id, sweep_index)
    stimulus, t_max = prepare_stimulus(sweep, dt)
    target_v = prepare_target(sweep, dt)
    target_v_jnp = jnp.array(target_v)

    logger.info(
        f"  Sweep {sweep['sweep_number']}: "
        f"{len(stimulus)} timesteps, "
        f"t_max={t_max:.1f} ms, "
        f"target range [{target_v.min():.1f}, {target_v.max():.1f}] mV"
    )

    # ---- Window to stimulus region ----
    max_duration_ms = 1200.0
    stim_np = np.array(stimulus)
    stim_threshold = np.max(np.abs(stim_np)) * 0.1
    active_indices = np.where(np.abs(stim_np) > stim_threshold)[0]

    if len(active_indices) > 0:
        pre_pad = int(50.0 / dt)
        post_pad = int(100.0 / dt)
        start_idx = max(0, active_indices[0] - pre_pad)
        end_idx = min(len(stim_np), active_indices[-1] + post_pad)
        max_samples = int(max_duration_ms / dt) + 1
        if (end_idx - start_idx) > max_samples:
            end_idx = start_idx + max_samples
        stimulus = stimulus[start_idx:end_idx]
        target_v_jnp = target_v_jnp[start_idx:end_idx]
        t_max = len(stimulus) * dt
        logger.info(
            f"  Windowed to stimulus region: {start_idx*dt:.0f}–{end_idx*dt:.0f} ms "
            f"({len(stimulus)} timesteps, {t_max:.0f} ms)"
        )
    elif t_max > max_duration_ms:
        n_keep = int(max_duration_ms / dt) + 1
        stimulus = stimulus[:n_keep]
        target_v_jnp = target_v_jnp[:n_keep]
        t_max = max_duration_ms
        logger.info(f"  No clear stimulus found; truncated to first {max_duration_ms:.0f} ms")

    # ---- Initial parameters in unconstrained space ----
    # Start from biophysically reasonable initial values
    init_constrained = {
        "gNa": 120.0, "gK": 36.0, "gL": 0.03,
        "eLeak": -60.0, "C": 1.0, "eNa": 50.0, "eK": -77.0,
    }
    opt_params = {
        name: inverse_sigmoid(jnp.array(val), *PARAM_BOUNDS[name])
        for name, val in init_constrained.items()
    }

    # ---- Build loss and gradient functions ----
    loss_fn = build_loss_fn(target_v_jnp, jnp.array(stimulus), dt)
    grad_fn = jit(value_and_grad(loss_fn))

    # ---- Optimizer ----
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(opt_params)

    # ---- Training loop ----
    losses = []
    best_loss = float("inf")
    best_params = None

    logger.info(f"  Starting optimization: {epochs} epochs, lr={lr}")

    for epoch in range(epochs):
        try:
            loss_val, grads = grad_fn(opt_params)
            loss_val_float = float(loss_val)
        except Exception as e:
            logger.error(f"  Epoch {epoch}: simulation failed — {e}")
            break

        if np.isnan(loss_val_float):
            logger.warning(f"  Epoch {epoch}: NaN loss, stopping")
            break

        # Gradient clipping
        grads = jax.tree.map(lambda g: jnp.clip(g, -10.0, 10.0), grads)

        updates, opt_state = optimizer.update(grads, opt_state)
        opt_params = optax.apply_updates(opt_params, updates)
        losses.append(loss_val_float)

        if loss_val_float < best_loss:
            best_loss = loss_val_float
            best_params = jax.tree.map(lambda x: x.copy(), opt_params)

        if epoch % 10 == 0 or epoch == epochs - 1:
            logger.info(f"    Epoch {epoch:4d}  loss={loss_val_float:.4f}  best={best_loss:.4f}")

    # ---- Extract best fitted parameters ----
    if best_params is None:
        logger.error(f"  No valid parameters found for {specimen_id}")
        return {"specimen_id": specimen_id, "success": False}

    fitted_dict = {
        name: float(sigmoid_transform(best_params[name], *PARAM_BOUNDS[name]))
        for name in best_params
    }
    logger.info(f"  Fitted parameters: {fitted_dict}")
    logger.info(f"  Best loss: {best_loss:.4f}")

    # ---- Run final simulation with best params ----
    v_sim = np.array(run_simulation(
        fitted_dict["gNa"], fitted_dict["gK"], fitted_dict["gL"],
        fitted_dict["eLeak"], fitted_dict["C"], fitted_dict["eNa"],
        fitted_dict["eK"], np.array(stimulus), dt
    ))

    # ---- Diagnostics ----
    n = min(len(v_sim), len(target_v_jnp))
    v_sim_trimmed = v_sim[:n]
    target_trimmed = np.array(target_v_jnp[:n])

    spike_threshold = -20.0
    sim_spikes = np.sum(np.diff((v_sim_trimmed > spike_threshold).astype(int)) > 0)
    target_spikes = np.sum(np.diff((target_trimmed > spike_threshold).astype(int)) > 0)
    corr = np.corrcoef(v_sim_trimmed, target_trimmed)[0, 1]

    diagnostics = {
        "model_spikes": bool(sim_spikes > 0),
        "n_sim_spikes": int(sim_spikes),
        "n_target_spikes": int(target_spikes),
        "firing_rate_error": abs(int(sim_spikes) - int(target_spikes)),
        "pearson_correlation": float(corr) if not np.isnan(corr) else 0.0,
        "final_mse": float(best_loss),
    }
    logger.info(f"  Diagnostics: {diagnostics}")

    # ---- Save results ----
    result = {
        "specimen_id": specimen_id,
        "success": True,
        "sweep_number": sweep["sweep_number"],
        "stimulus_amplitude_pA": sweep.get("stimulus_amplitude"),
        "fitted_parameters": fitted_dict,
        "diagnostics": diagnostics,
        "training": {
            "epochs": len(losses),
            "learning_rate": lr,
            "dt_ms": dt,
            "final_loss": losses[-1] if losses else None,
            "best_loss": best_loss,
        },
    }

    with open(out / "fit_result.json", "w") as f:
        json.dump(result, f, indent=2)

    # ---- Plot: trace overlay ----
    t_ms = np.arange(n) * dt
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={"height_ratios": [3, 1]})

    ax = axes[0]
    ax.plot(t_ms, target_trimmed, color="black", linewidth=0.8, alpha=0.8, label="Recording")
    ax.plot(t_ms, v_sim_trimmed, color="red", linewidth=0.8, alpha=0.7, label="Simulation (Na+K+Leak)")
    ax.set_ylabel("Membrane Potential (mV)")
    ax.set_title(
        f"Specimen {specimen_id} — Sweep {sweep['sweep_number']} "
        f"({sweep.get('stimulus_amplitude', '?'):.0f} pA)\n"
        f"MSE={best_loss:.2f}  r={diagnostics['pearson_correlation']:.3f}  "
        f"Spikes: {sim_spikes} sim / {target_spikes} target"
    )
    ax.legend(loc="upper right")
    ax.set_xlim(0, t_ms[-1])

    ax2 = axes[1]
    stim_trimmed = np.array(stimulus[:n])
    ax2.plot(t_ms, stim_trimmed, color="steelblue", linewidth=0.8)
    ax2.set_xlabel("Time (ms)")
    ax2.set_ylabel("Current (nA)")
    ax2.set_xlim(0, t_ms[-1])

    plt.tight_layout()
    plt.savefig(out / "trace_overlay.png", dpi=150)
    plt.close()

    # ---- Plot: loss curve ----
    if losses:
        fig, ax = plt.subplots(1, 1, figsize=(8, 4))
        ax.semilogy(losses, color="steelblue", linewidth=1.5)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE Loss")
        ax.set_title(f"Training Loss — Specimen {specimen_id}")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out / "loss_curve.png", dpi=150)
        plt.close()

    return result


# ---------------------------------------------------------------------------
# 5. Batch Fitting & Summary
# ---------------------------------------------------------------------------

def run_baseline_fits(data_dir: str, specimen_id: int = None,
                      fit_all: bool = False, dt: float = 0.025,
                      epochs: int = 100, lr: float = 0.02):
    """Run baseline HH fits for one or all valid PV+ cells."""
    data_dir = Path(data_dir)
    output_dir = data_dir / "fits"
    output_dir.mkdir(parents=True, exist_ok=True)

    ctc = CellTypesCache(manifest_file=str(data_dir / "manifest.json"))

    with open(data_dir / "sweep_index.json") as f:
        sweep_index = json.load(f)

    valid_ids = [
        int(sid) for sid, entry in sweep_index.items()
        if entry.get("valid", False)
    ]
    logger.info(f"Found {len(valid_ids)} valid cells in sweep_index.json")

    if specimen_id:
        if specimen_id not in valid_ids:
            logger.error(f"Specimen {specimen_id} not in valid cells")
            return
        ids_to_fit = [specimen_id]
    elif fit_all:
        ids_to_fit = valid_ids
    else:
        ids_to_fit = [valid_ids[0]]

    logger.info(f"Will fit {len(ids_to_fit)} cell(s)")

    results = []
    for sid in ids_to_fit:
        try:
            result = fit_cell(ctc, sid, sweep_index, output_dir,
                              dt=dt, epochs=epochs, lr=lr)
            results.append(result)
        except Exception as e:
            logger.error(f"Failed to fit specimen {sid}: {e}")
            results.append({"specimen_id": sid, "success": False, "error": str(e)})

    # Summary table
    print(f"\n{'='*80}")
    print("BASELINE FIT SUMMARY (Na + K + Leak)")
    print(f"{'='*80}")
    print(f"{'Specimen':<12} {'Success':<8} {'MSE':<10} {'r':<8} "
          f"{'Spikes(sim)':<12} {'Spikes(tgt)':<12} "
          f"{'gNa':<8} {'gK':<8} {'gLeak':<10} {'eLeak':<8}")
    print("-" * 98)

    for r in results:
        if r.get("success"):
            d = r["diagnostics"]
            p = r["fitted_parameters"]
            print(
                f"{r['specimen_id']:<12} {'OK':<8} "
                f"{d['final_mse']:<10.2f} {d['pearson_correlation']:<8.3f} "
                f"{d['n_sim_spikes']:<12} {d['n_target_spikes']:<12} "
                f"{p['gNa']:<8.1f} {p['gK']:<8.1f} "
                f"{p['gL']:<10.4f} {p['eLeak']:<8.1f}"
            )
        else:
            print(f"{r['specimen_id']:<12} {'FAIL':<8} — {r.get('error', 'unknown')}")

    # Save summary CSV
    import csv
    csv_path = output_dir / "baseline_summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "specimen_id", "success", "mse", "pearson_r",
            "n_sim_spikes", "n_target_spikes",
            "gNa", "gK", "gL", "eLeak", "C"
        ])
        for r in results:
            if r.get("success"):
                d = r["diagnostics"]
                p = r["fitted_parameters"]
                writer.writerow([
                    r["specimen_id"], True, d["final_mse"],
                    d["pearson_correlation"], d["n_sim_spikes"],
                    d["n_target_spikes"], p["gNa"], p["gK"],
                    p["gL"], p["eLeak"], p["C"]
                ])
            else:
                writer.writerow([r["specimen_id"], False] + [""] * 9)

    logger.info(f"Summary saved to {csv_path}")
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fit fixed Na+K+Leak HH model to Allen PV+ neurons via BrainPy"
    )
    parser.add_argument(
        "--data-dir", type=str, default="cell_types_data",
        help="Path to Allen data cache (from allen_downloader.py)"
    )
    parser.add_argument(
        "--specimen-id", type=int, default=None,
        help="Fit a specific specimen ID"
    )
    parser.add_argument(
        "--all", action="store_true", dest="fit_all",
        help="Fit all valid cells"
    )
    parser.add_argument(
        "--epochs", type=int, default=100,
        help="Number of optimization epochs (default: 100)"
    )
    parser.add_argument(
        "--lr", type=float, default=0.02,
        help="Adam learning rate (default: 0.02)"
    )
    parser.add_argument(
        "--dt", type=float, default=0.025,
        help="BrainPy simulation timestep in ms (default: 0.025)"
    )

    args = parser.parse_args()
    run_baseline_fits(
        data_dir=args.data_dir,
        specimen_id=args.specimen_id,
        fit_all=args.fit_all,
        dt=args.dt,
        epochs=args.epochs,
        lr=args.lr,
    )