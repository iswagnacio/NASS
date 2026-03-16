"""
Jaxley Baseline Fitter — Fixed Na+K+Leak HH Model (Weeks 3–4)
==============================================================

Fits a single-compartment Hodgkin-Huxley model (Na + K + Leak) to PV+
fast-spiking interneuron recordings from the Allen Cell Types Database
using Jaxley's differentiable simulation and gradient descent.

This is the "Jaxley fixed-HH" baseline from the NASS proposal — the
simplest biophysical model that the agent-discovered models must beat.

Usage:
    python jaxley_fit.py --data-dir ./cell_types_data
    python jaxley_fit.py --data-dir ./cell_types_data --specimen-id 469801138
    python jaxley_fit.py --data-dir ./cell_types_data --epochs 200 --lr 0.01
    python jaxley_fit.py --data-dir ./cell_types_data --all  # fit all valid cells

Requires:
    pip install jaxley jax jaxlib optax allensdk matplotlib

Output:
    {data_dir}/fits/
    ├── {specimen_id}/
    │   ├── fit_result.json      — fitted parameters, losses, diagnostics
    │   ├── trace_overlay.png    — simulated vs recorded voltage traces
    │   └── loss_curve.png       — training loss over epochs
    └── baseline_summary.csv     — comparison table across all fitted cells
"""

# ---- JAX config must come before any JAX imports ----
from jax import config
config.update("jax_enable_x64", False)
config.update("jax_platform_name", "METAL")

import os
import json
import argparse
import logging
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, value_and_grad
import optax
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import jaxley as jx
from jaxley.channels import Na, K, Leak

from allensdk.core.cell_types_cache import CellTypesCache

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Allen Data Loader — extract a single training sweep
# ---------------------------------------------------------------------------

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

    # Sort by amplitude
    by_amp = sorted(train_sweeps, key=lambda s: s.get("stimulus_amplitude", 0) or 0)

    # Find sweeps with spikes (num_spikes > 0)
    spiking = [sw for sw in by_amp if (sw.get("num_spikes") or 0) > 0]

    if not spiking:
        # num_spikes metadata may be missing — check spike_times from NWB directly
        # for the top few highest-amplitude sweeps
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
        # Re-sort spiking sweeps by amplitude
        spiking.sort(key=lambda s: s.get("stimulus_amplitude", 0) or 0)

    if spiking:
        # Pick a sweep ~1/3 into the spiking range (above rheobase but
        # not so high that we risk depolarization block)
        idx = min(len(spiking) - 1, max(1, len(spiking) // 3))
        chosen = spiking[idx]
        logger.info(
            f"  Selected spiking sweep {chosen['sweep_number']} "
            f"(amplitude={chosen.get('stimulus_amplitude', '?'):.1f} pA, "
            f"num_spikes={chosen.get('num_spikes', '?')}, "
            f"index {idx}/{len(spiking)} spiking sweeps)"
        )
    else:
        # Fallback: highest amplitude training sweep
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
    """
    Load multiple training sweeps spanning a range of amplitudes.
    Returns a list of sweep dicts.
    """
    cell_entry = sweep_index[str(specimen_id)]
    train_sweeps = cell_entry["split"]["training"]["long_square"]

    if not train_sweeps:
        raise ValueError(f"No training sweeps for specimen {specimen_id}")

    by_amp = sorted(train_sweeps, key=lambda s: s.get("stimulus_amplitude", 0) or 0)
    n = min(n_sweeps, len(by_amp))
    # Sample evenly across amplitude range
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
# 2. Jaxley Model Builder — single-compartment Na+K+Leak
# ---------------------------------------------------------------------------

def build_hh_cell(dt: float = 0.025) -> jx.Compartment:
    """
    Build a single-compartment HH cell with Na, K, and Leak channels.
    This is the fixed 3-channel baseline model.

    Geometry is set to approximate a PV+ fast-spiking interneuron soma
    (~20 um diameter sphere → single compartment with equivalent area).
    Conductance densities are initialised higher than squid-axon HH
    to reflect the high channel density of fast-spiking cells.
    """
    comp = jx.Compartment()
    comp.insert(Na())
    comp.insert(K())
    comp.insert(Leak())

    # Soma geometry: ~20 um diameter sphere
    # For a sphere of diameter d, surface area = pi*d^2
    # Jaxley compartment area = 2*pi*radius*length
    # Set radius=10, length=10*pi ≈ 31.4 so area ≈ pi*(20)^2 ≈ 1257 um^2
    comp.set("radius", 10.0)           # um
    comp.set("length", 31.4)           # um (gives sphere-like area)
    comp.set("axial_resistivity", 100.0)  # ohm*cm
    comp.set("capacitance", 1.0)       # uF/cm^2

    # Channel conductances — PV+ FS interneurons have very high Na/K density
    comp.set("Na_gNa", 0.5)           # S/cm^2 (high for FS cells)
    comp.set("K_gK", 0.2)             # S/cm^2 (high for FS cells)
    comp.set("Leak_gLeak", 0.001)     # S/cm^2

    # Reversal potentials (eNa and eK are global, not channel-prefixed)
    comp.set("eNa", 50.0)             # mV
    comp.set("eK", -77.0)             # mV
    comp.set("Leak_eLeak", -60.0)     # mV

    return comp


def prepare_stimulus(sweep: dict, dt: float = 0.025) -> np.ndarray:
    """
    Resample the Allen stimulus waveform to Jaxley's simulation timestep.
    Allen recordings are typically at 200 kHz; Jaxley uses dt in ms.

    Allen stimulus is in Amps; Jaxley expects nA.
    """
    sr = sweep["sampling_rate"]
    stimulus_amps = sweep["stimulus"]

    # Convert A -> nA
    stimulus_nA = stimulus_amps * 1e9

    # Compute time vectors
    t_allen = np.arange(len(stimulus_nA)) / sr  # seconds
    t_max_s = t_allen[-1]
    t_max_ms = t_max_s * 1000.0

    # Jaxley time in ms
    n_steps = int(t_max_ms / dt) + 1
    t_jaxley_ms = np.arange(n_steps) * dt
    t_jaxley_s = t_jaxley_ms / 1000.0

    # Resample stimulus to Jaxley timestep via interpolation
    stimulus_resampled = np.interp(t_jaxley_s, t_allen, stimulus_nA)

    return stimulus_resampled, t_max_ms


def prepare_target(sweep: dict, dt: float = 0.025) -> np.ndarray:
    """
    Resample the Allen voltage response to Jaxley's simulation timestep.
    Allen response is in Volts; Jaxley uses mV.
    """
    sr = sweep["sampling_rate"]
    response_V = sweep["response"]

    # Convert V -> mV
    response_mV = response_V * 1e3

    t_allen = np.arange(len(response_V)) / sr  # seconds
    t_max_s = t_allen[-1]
    t_max_ms = t_max_s * 1000.0

    n_steps = int(t_max_ms / dt) + 1
    t_jaxley_ms = np.arange(n_steps) * dt
    t_jaxley_s = t_jaxley_ms / 1000.0

    response_resampled = np.interp(t_jaxley_s, t_allen, response_mV)

    return response_resampled


# ---------------------------------------------------------------------------
# 3. Simulation & Loss Functions
# ---------------------------------------------------------------------------

def setup_simulation(cell: jx.Compartment, stimulus: np.ndarray,
                     dt: float = 0.025, t_max: float = None):
    """
    Configure stimulus injection and recording on the cell.
    stimulus: 1D array of current values in nA, one per Jaxley timestep.

    Uses cell.stimulate() which registers the stimulus persistently,
    so jx.integrate() knows the simulation duration.
    """
    cell.delete_stimuli()
    cell.delete_recordings()

    # Inject the full stimulus waveform using .stimulate()
    # stimulus must be shape (n_timesteps,) — Jaxley infers duration from it
    i_ext = jnp.array(stimulus)
    cell.stimulate(i_ext)

    # Record membrane voltage
    cell.record("v")

    return cell


def build_loss_fn(cell, target_v, dt, transform):
    """
    Build a differentiable loss function that:
    1. Takes optimizable parameters
    2. Sets them on the cell via data_set
    3. Runs Jaxley simulation
    4. Computes MSE between simulated and recorded voltage

    The loss combines:
    - Waveform MSE (primary)
    - Penalty for non-physiological resting potential
    """

    def loss_fn(opt_params):
        # Transform from unconstrained -> constrained parameter space
        params = transform.forward(opt_params)

        # Set parameters on the cell
        param_state = None
        param_state = cell.data_set("Na_gNa", params[0]["Na_gNa"], param_state)
        param_state = cell.data_set("K_gK", params[1]["K_gK"], param_state)
        param_state = cell.data_set("Leak_gLeak", params[2]["Leak_gLeak"], param_state)
        param_state = cell.data_set("Leak_eLeak", params[3]["Leak_eLeak"], param_state)
        param_state = cell.data_set("capacitance", params[4]["capacitance"], param_state)
        param_state = cell.data_set("eNa", params[5]["eNa"], param_state)
        param_state = cell.data_set("eK", params[6]["eK"], param_state)
        param_state = cell.data_set("radius", params[7]["radius"], param_state)

        # Run simulation
        v = jx.integrate(cell, param_state=param_state, delta_t=dt)
        v_sim = v[0]  # shape: (n_timesteps,)

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

    # ---- Find the active stimulus window and extract around it ----
    # Allen long-square traces are ~5s with the current step starting ~1s in.
    # We need to include the stimulus period, not just the pre-stimulus baseline.
    # Strategy: find where stimulus is significantly non-zero, then take a window
    # starting slightly before stimulus onset.
    max_duration_ms = 1200.0  # enough for stimulus + some baseline
    stim_np = np.array(stimulus)
    stim_threshold = np.max(np.abs(stim_np)) * 0.1  # 10% of peak
    active_indices = np.where(np.abs(stim_np) > stim_threshold)[0]

    if len(active_indices) > 0:
        # Start 50ms before stimulus onset, end 100ms after stimulus offset
        pre_pad = int(50.0 / dt)
        post_pad = int(100.0 / dt)
        start_idx = max(0, active_indices[0] - pre_pad)
        end_idx = min(len(stim_np), active_indices[-1] + post_pad)

        # Cap total duration
        max_samples = int(max_duration_ms / dt) + 1
        if (end_idx - start_idx) > max_samples:
            end_idx = start_idx + max_samples

        stimulus = stimulus[start_idx:end_idx]
        target_v_jnp = target_v_jnp[start_idx:end_idx]
        t_max = len(stimulus) * dt
        logger.info(
            f"  Windowed to stimulus region: {start_idx*dt:.0f}–{end_idx*dt:.0f} ms "
            f"({len(stimulus)} timesteps, {t_max:.0f} ms), "
            f"target range [{float(target_v_jnp.min()):.1f}, {float(target_v_jnp.max()):.1f}] mV"
        )
    elif t_max > max_duration_ms:
        n_keep = int(max_duration_ms / dt) + 1
        stimulus = stimulus[:n_keep]
        target_v_jnp = target_v_jnp[:n_keep]
        t_max = max_duration_ms
        logger.info(f"  No clear stimulus found; truncated to first {max_duration_ms:.0f} ms")

    # ---- Build model ----
    cell = build_hh_cell(dt)
    cell = setup_simulation(cell, stimulus, dt, t_max)

    # ---- Make parameters trainable with bounds ----
    # Conductances (S/cm^2), reversal (mV), capacitance (uF/cm^2)
    cell.make_trainable("Na_gNa")
    cell.make_trainable("K_gK")
    cell.make_trainable("Leak_gLeak")
    cell.make_trainable("Leak_eLeak")
    cell.make_trainable("capacitance")
    cell.make_trainable("eNa")
    cell.make_trainable("eK")
    cell.make_trainable("radius")

    opt_params = cell.get_parameters()

    # Sigmoid transforms to keep parameters in biophysical bounds
    from jaxley.optimize.transforms import ParamTransform, SigmoidTransform

    transforms = [
        {"Na_gNa": SigmoidTransform(lower=0.05, upper=5.0)},    # FS cells can be very high
        {"K_gK": SigmoidTransform(lower=0.01, upper=2.0)},      # also high for FS
        {"Leak_gLeak": SigmoidTransform(lower=1e-5, upper=0.05)},
        {"Leak_eLeak": SigmoidTransform(lower=-80.0, upper=-40.0)},
        {"capacitance": SigmoidTransform(lower=0.5, upper=3.0)},
        {"eNa": SigmoidTransform(lower=30.0, upper=70.0)},
        {"eK": SigmoidTransform(lower=-100.0, upper=-60.0)},
        {"radius": SigmoidTransform(lower=3.0, upper=30.0)},    # controls effective area
    ]
    transform = ParamTransform(transforms)

    # ---- Build loss and gradient functions ----
    loss_fn = build_loss_fn(cell, target_v_jnp, dt, transform)
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

        # Check for NaN
        if np.isnan(loss_val_float):
            logger.warning(f"  Epoch {epoch}: NaN loss, stopping")
            break

        # Gradient clipping to prevent instability
        grads = jax.tree.map(
            lambda g: jnp.clip(g, -10.0, 10.0), grads
        )

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

    fitted = transform.forward(best_params)
    fitted_dict = {
        "Na_gNa": float(fitted[0]["Na_gNa"][0]),
        "K_gK": float(fitted[1]["K_gK"][0]),
        "Leak_gLeak": float(fitted[2]["Leak_gLeak"][0]),
        "Leak_eLeak": float(fitted[3]["Leak_eLeak"][0]),
        "capacitance": float(fitted[4]["capacitance"][0]),
        "eNa": float(fitted[5]["eNa"][0]),
        "eK": float(fitted[6]["eK"][0]),
        "radius": float(fitted[7]["radius"][0]),
    }

    logger.info(f"  Fitted parameters: {fitted_dict}")
    logger.info(f"  Best loss: {best_loss:.4f}")

    # ---- Run final simulation with best params ----
    param_state = None
    param_state = cell.data_set("Na_gNa", fitted[0]["Na_gNa"], param_state)
    param_state = cell.data_set("K_gK", fitted[1]["K_gK"], param_state)
    param_state = cell.data_set("Leak_gLeak", fitted[2]["Leak_gLeak"], param_state)
    param_state = cell.data_set("Leak_eLeak", fitted[3]["Leak_eLeak"], param_state)
    param_state = cell.data_set("capacitance", fitted[4]["capacitance"], param_state)
    param_state = cell.data_set("eNa", fitted[5]["eNa"], param_state)
    param_state = cell.data_set("eK", fitted[6]["eK"], param_state)
    param_state = cell.data_set("radius", fitted[7]["radius"], param_state)

    v_final = jx.integrate(cell, param_state=param_state, delta_t=dt)
    v_sim = np.array(v_final[0])

    # ---- Diagnostics ----
    n = min(len(v_sim), len(target_v_jnp))
    v_sim_trimmed = v_sim[:n]
    target_trimmed = np.array(target_v_jnp[:n])

    # Does the model spike?
    spike_threshold = -20.0  # mV
    sim_spikes = np.sum(np.diff((v_sim_trimmed > spike_threshold).astype(int)) > 0)
    target_spikes = np.sum(np.diff((target_trimmed > spike_threshold).astype(int)) > 0)

    # Correlation
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

    # Stimulus trace
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
    """
    Run baseline HH fits for one or all valid PV+ cells.
    """
    data_dir = Path(data_dir)
    output_dir = data_dir / "fits"
    output_dir.mkdir(parents=True, exist_ok=True)

    ctc = CellTypesCache(manifest_file=str(data_dir / "manifest.json"))

    with open(data_dir / "sweep_index.json") as f:
        sweep_index = json.load(f)

    # Get valid cell IDs
    valid_ids = [
        int(sid) for sid, entry in sweep_index.items()
        if entry.get("valid", False)
    ]
    logger.info(f"Found {len(valid_ids)} valid cells in sweep_index.json")

    # Select which cells to fit
    if specimen_id:
        if specimen_id not in valid_ids:
            logger.error(f"Specimen {specimen_id} not in valid cells")
            return
        ids_to_fit = [specimen_id]
    elif fit_all:
        ids_to_fit = valid_ids
    else:
        # Default: fit the first valid cell
        ids_to_fit = [valid_ids[0]]

    logger.info(f"Will fit {len(ids_to_fit)} cell(s)")

    # Run fits
    results = []
    for sid in ids_to_fit:
        try:
            result = fit_cell(
                ctc, sid, sweep_index, output_dir,
                dt=dt, epochs=epochs, lr=lr
            )
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
                f"{p['Na_gNa']:<8.4f} {p['K_gK']:<8.4f} "
                f"{p['Leak_gLeak']:<10.6f} {p['Leak_eLeak']:<8.1f}"
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
            "Na_gNa", "K_gK", "Leak_gLeak", "Leak_eLeak", "capacitance"
        ])
        for r in results:
            if r.get("success"):
                d = r["diagnostics"]
                p = r["fitted_parameters"]
                writer.writerow([
                    r["specimen_id"], True, d["final_mse"],
                    d["pearson_correlation"], d["n_sim_spikes"],
                    d["n_target_spikes"], p["Na_gNa"], p["K_gK"],
                    p["Leak_gLeak"], p["Leak_eLeak"], p["capacitance"]
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
        description="Fit fixed Na+K+Leak HH model to Allen PV+ neurons via Jaxley"
    )
    parser.add_argument(
        "--data-dir", type=str, default="cell_types_data",
        help="Path to Allen data cache (from allen_pv_pipeline.py)"
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
        help="Adam learning rate (default: 0.005)"
    )
    parser.add_argument(
        "--dt", type=float, default=0.025,
        help="Jaxley simulation timestep in ms (default: 0.025)"
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