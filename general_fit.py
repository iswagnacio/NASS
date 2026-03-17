"""
Generalized Jaxley Fitter — Fit Any Agent-Proposed Model (Weeks 9–10)
=====================================================================

Bridges sga.py's ModelProposal with jaxley_fit.py's fitting infrastructure.
Instead of hardcoding Na+K+Leak, this module dynamically builds a Jaxley
compartment from whatever channels the LLM agent proposes, sets up the
appropriate trainable parameters and bounds, runs gradient descent, and
returns a DiagnosticReport the outer loop can use.

This is the integration piece that replaces sga.py's placeholder
_run_inner_loop() with real Jaxley execution.

Usage:
    from generalized_fit import fit_proposal
    from sga import ModelProposal, DiagnosticReport

    proposal = ModelProposal(
        channels=["Na", "K", "Leak", "Kv3"],
        param_config={
            "Na_gNa":  {"init": 0.5, "lower": 0.05, "upper": 5.0},
            "K_gK":    {"init": 0.2, "lower": 0.01, "upper": 2.0},
            "Kv3_gKv3":{"init": 0.01, "lower": 1e-4, "upper": 0.1},
        },
    )
    report = fit_proposal(proposal, specimen_id=509683388, data_dir="./data")

Requires:
    pip install jaxley jax jaxlib optax allensdk
"""

# ---- JAX config must come before any JAX imports ----
from jax import config
config.update("jax_enable_x64", False)

import json
import logging
import traceback
from pathlib import Path
from typing import Optional

import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, value_and_grad
import optax

import jaxley as jx
from jaxley.channels import Na, K, Leak
from jaxley.optimize.transforms import ParamTransform, SigmoidTransform

from allensdk.core.cell_types_cache import CellTypesCache

from channels import Kv3, IM, IAHP, IT, ICaL, IH, CHANNEL_REGISTRY
from sga import ModelProposal, DiagnosticReport
from sim_fit import (
    load_training_sweep,
    prepare_stimulus,
    prepare_target,
    setup_simulation,
)

logger = logging.getLogger(__name__)


# ===========================================================================
# Channel Resolution — map names to classes
# ===========================================================================

BUILTIN_CHANNELS = {
    "Na": Na,
    "K": K,
    "Leak": Leak,
}

CUSTOM_CHANNELS = {
    name: info["class"] for name, info in CHANNEL_REGISTRY.items()
}

ALL_CHANNELS = {**BUILTIN_CHANNELS, **CUSTOM_CHANNELS}

# Default parameter bounds per channel when the proposal doesn't specify them.
# These are biophysically reasonable ranges from the proposal + Pospischil (2008).
# Bounds are deliberately tighter than the full biophysical range to prevent
# the optimizer from "cheating" (e.g. inflating radius to dilute current,
# shifting reversal potentials to extremes instead of learning to spike).
DEFAULT_PARAM_BOUNDS = {
    # Conductances — wide enough for FS cells
    "Na_gNa":       {"init": 0.5,   "lower": 0.05,  "upper": 15.0},
    "K_gK":         {"init": 0.2,   "lower": 0.01,  "upper": 2.0},
    "Leak_gLeak":   {"init": 0.001, "lower": 1e-5,  "upper": 0.01},
    "Leak_eLeak":   {"init": -65.0, "lower": -75.0, "upper": -50.0},
    # Custom channels
    "Kv3_gKv3":     {"init": 0.01,  "lower": 1e-4,  "upper": 0.1},
    "IM_gM":        {"init": 1e-4,  "lower": 1e-6,  "upper": 1e-3},
    "IAHP_gAHP":    {"init": 1e-4,  "lower": 1e-6,  "upper": 1e-3},
    "IT_gT":        {"init": 1e-4,  "lower": 1e-5,  "upper": 1e-2},
    "ICaL_gCaL":    {"init": 1e-4,  "lower": 1e-5,  "upper": 1e-2},
    "IH_gH":        {"init": 1e-5,  "lower": 1e-6,  "upper": 1e-3},
    # Reversal potentials — tighter to prevent optimizer cheating
    "eNa":          {"init": 50.0,  "lower": 40.0,  "upper": 70.0},
    "eK":           {"init": -77.0, "lower": -90.0, "upper": -70.0},
    # Geometry — tighter radius prevents current dilution
    "capacitance":  {"init": 1.0,   "lower": 0.5,   "upper": 2.0},
    "radius":       {"init": 10.0,  "lower": 5.0,   "upper": 15.0},
}

# Map channel names to their conductance parameter names.
# This is needed so we know what to make trainable for each channel.
CHANNEL_CONDUCTANCE_PARAMS = {
    "Na":   ["Na_gNa"],
    "K":    ["K_gK"],
    "Leak": ["Leak_gLeak", "Leak_eLeak"],
    "Kv3":  ["Kv3_gKv3"],
    "IM":   ["IM_gM"],
    "IAHP": ["IAHP_gAHP"],
    "IT":   ["IT_gT"],
    "ICaL": ["ICaL_gCaL"],
    "IH":   ["IH_gH"],
}

# Global parameters always made trainable
GLOBAL_TRAINABLE = ["eNa", "eK", "capacitance", "radius"]


# ===========================================================================
# Build Cell from Proposal
# ===========================================================================

def build_cell_from_proposal(proposal: ModelProposal) -> tuple:
    """
    Dynamically build a Jaxley single-compartment cell from a ModelProposal.

    Returns:
        (cell, trainable_params, error_message)
        If error_message is not None, cell construction failed.
    """
    # Validate channel names
    invalid = [ch for ch in proposal.channels if ch not in ALL_CHANNELS]
    if invalid:
        return None, None, f"Unknown channels: {invalid}. Available: {list(ALL_CHANNELS.keys())}"

    # Ensure Na, K, Leak are always present (biophysical minimum)
    channels = list(proposal.channels)
    for required in ["Na", "K", "Leak"]:
        if required not in channels:
            channels.insert(0, required)
            logger.info(f"  Auto-added required channel: {required}")

    try:
        comp = jx.Compartment()

        # Insert channels
        for ch_name in channels:
            ch_class = ALL_CHANNELS[ch_name]
            comp.insert(ch_class())

        # Set geometry
        comp.set("radius", proposal.radius)
        comp.set("length", proposal.length)
        comp.set("capacitance", proposal.capacitance)
        comp.set("axial_resistivity", 100.0)

        # Set reversal potentials
        comp.set("eNa", 50.0)
        comp.set("eK", -77.0)

        # Set initial conductance values from proposal or defaults
        for ch_name in channels:
            for param_name in CHANNEL_CONDUCTANCE_PARAMS.get(ch_name, []):
                if param_name in proposal.param_config:
                    init_val = proposal.param_config[param_name].get("init")
                elif param_name in DEFAULT_PARAM_BOUNDS:
                    init_val = DEFAULT_PARAM_BOUNDS[param_name]["init"]
                else:
                    continue
                try:
                    comp.set(param_name, init_val)
                except Exception as e:
                    logger.warning(f"  Could not set {param_name}={init_val}: {e}")

    except Exception as e:
        return None, None, f"Cell construction failed: {e}\n{traceback.format_exc()}"

    # Collect the list of trainable parameter names + their bounds
    trainable = []

    for ch_name in channels:
        for param_name in CHANNEL_CONDUCTANCE_PARAMS.get(ch_name, []):
            # Get bounds from proposal, then defaults
            if param_name in proposal.param_config:
                cfg = proposal.param_config[param_name]
            elif param_name in DEFAULT_PARAM_BOUNDS:
                cfg = DEFAULT_PARAM_BOUNDS[param_name]
            else:
                continue
            trainable.append({
                "name": param_name,
                "lower": cfg.get("lower", cfg["init"] * 0.01),
                "upper": cfg.get("upper", cfg["init"] * 100.0),
            })

    for param_name in GLOBAL_TRAINABLE:
        if param_name in DEFAULT_PARAM_BOUNDS:
            cfg = proposal.param_config.get(param_name, DEFAULT_PARAM_BOUNDS[param_name])
            trainable.append({
                "name": param_name,
                "lower": cfg.get("lower", DEFAULT_PARAM_BOUNDS[param_name]["lower"]),
                "upper": cfg.get("upper", DEFAULT_PARAM_BOUNDS[param_name]["upper"]),
            })

    return comp, trainable, None


# ===========================================================================
# Build Loss Function (generalized)
# ===========================================================================

def build_generalized_loss_fn(cell, target_v, dt, transform, param_names):
    """
    Build a differentiable loss function for an arbitrary set of trainable
    parameters. This generalizes jaxley_fit.py's build_loss_fn() which
    hardcoded Na_gNa, K_gK, etc.
    """
    def loss_fn(opt_params):
        params = transform.forward(opt_params)

        param_state = None
        for i, name in enumerate(param_names):
            param_state = cell.data_set(name, params[i][name], param_state)

        try:
            v = jx.integrate(cell, param_state=param_state, delta_t=dt)
            v_sim = v[0]

            n = min(len(v_sim), len(target_v))
            mse = jnp.mean((v_sim[:n] - target_v[:n]) ** 2)
        except Exception:
            mse = jnp.inf

        return mse

    return loss_fn


# ===========================================================================
# Stimulus Windowing
# ===========================================================================

def window_to_main_stimulus(stimulus, target_v_jnp, dt, max_duration_ms=1200.0):
    """
    Window stimulus and target to the main current step, ignoring short
    test pulses. Finds the longest contiguous block of supra-threshold
    stimulus and pads around it.

    Returns (stimulus_windowed, target_windowed, t_max_ms).
    """
    stim_np = np.array(stimulus)
    stim_threshold = np.max(np.abs(stim_np)) * 0.1
    above = np.abs(stim_np) > stim_threshold
    active_indices = np.where(above)[0]

    if len(active_indices) > 0:
        # Split into contiguous blocks and find the longest one
        breaks = np.where(np.diff(active_indices) > int(5.0 / dt))[0]
        if len(breaks) > 0:
            blocks = np.split(active_indices, breaks + 1)
            main_block = max(blocks, key=len)
        else:
            main_block = active_indices

        pre_pad = int(50.0 / dt)
        post_pad = int(100.0 / dt)
        start_idx = max(0, main_block[0] - pre_pad)
        end_idx = min(len(stim_np), main_block[-1] + post_pad)

        max_samples = int(max_duration_ms / dt) + 1
        if (end_idx - start_idx) > max_samples:
            end_idx = start_idx + max_samples

        stimulus = stimulus[start_idx:end_idx]
        target_v_jnp = target_v_jnp[start_idx:end_idx]
        t_max = len(stimulus) * dt

        logger.info(
            f"  Windowed to main stimulus: [{start_idx*dt:.0f}–{end_idx*dt:.0f}] ms, "
            f"{len(stimulus)} steps, {t_max:.0f} ms")
    elif len(stimulus) * dt > max_duration_ms:
        n_keep = int(max_duration_ms / dt) + 1
        stimulus = stimulus[:n_keep]
        target_v_jnp = target_v_jnp[:n_keep]
        t_max = max_duration_ms
        logger.info(f"  No clear stimulus found; truncated to {max_duration_ms:.0f} ms")
    else:
        t_max = len(stimulus) * dt

    return stimulus, target_v_jnp, t_max


# ===========================================================================
# Compute Diagnostics
# ===========================================================================

def compute_diagnostics(
    v_sim: np.ndarray,
    target_v: np.ndarray,
    dt: float,
    proposal: ModelProposal,
    fitted_params: dict,
    trainable: list,
    final_loss: float,
) -> dict:
    """
    Compute all diagnostic flags that the outer loop needs.
    Returns a dict suitable for constructing a DiagnosticReport.
    """
    n = min(len(v_sim), len(target_v))
    v_sim = v_sim[:n]
    target_v = target_v[:n]

    spike_threshold = -20.0

    sim_crossings = np.diff((v_sim > spike_threshold).astype(int)) > 0
    tgt_crossings = np.diff((target_v > spike_threshold).astype(int)) > 0
    n_sim_spikes = int(np.sum(sim_crossings))
    n_tgt_spikes = int(np.sum(tgt_crossings))

    corr = float(np.corrcoef(v_sim, target_v)[0, 1]) if len(v_sim) > 1 else 0.0
    if np.isnan(corr):
        corr = 0.0

    model_spikes = n_sim_spikes > 0

    # Diagnostic flags (from proposal Section 3.3)
    no_spikes = (n_tgt_spikes > 0) and (n_sim_spikes == 0)

    wrong_firing_rate = False
    if n_tgt_spikes > 0 and n_sim_spikes > 0:
        rate_ratio = n_sim_spikes / n_tgt_spikes
        wrong_firing_rate = rate_ratio < 0.5 or rate_ratio > 2.0

    # Spike width check
    broad_spikes = False
    if n_sim_spikes >= 3 and n_tgt_spikes >= 3:
        sim_crossing_idxs = np.where(sim_crossings)[0]
        tgt_crossing_idxs = np.where(tgt_crossings)[0]

        if len(sim_crossing_idxs) >= 2 and len(tgt_crossing_idxs) >= 2:
            for idx in sim_crossing_idxs[:5]:
                if idx + int(2.0 / dt) < len(v_sim):
                    post_spike = v_sim[idx:idx + int(2.0 / dt)]
                    if np.all(post_spike > spike_threshold):
                        broad_spikes = True
                        break

    # Sag check
    excessive_sag = False
    subthreshold_mask = target_v < -70.0
    if subthreshold_mask.sum() > 10:
        tgt_hyp = target_v[subthreshold_mask]
        sim_hyp = v_sim[subthreshold_mask] if subthreshold_mask.sum() <= len(v_sim) else np.array([])
        if len(sim_hyp) > 0:
            if np.min(sim_hyp) < np.min(tgt_hyp) - 10.0:
                excessive_sag = True

    # Parameters at bounds
    params_at_bounds = []
    for t_info in trainable:
        name = t_info["name"]
        lower = t_info["lower"]
        upper = t_info["upper"]
        if name in fitted_params:
            val = fitted_params[name]
            margin = (upper - lower) * 0.02  # within 2% of bound
            if abs(val - lower) < margin:
                params_at_bounds.append(f"{name} (at lower={lower})")
            elif abs(val - upper) < margin:
                params_at_bounds.append(f"{name} (at upper={upper})")

    return {
        "final_loss": final_loss,
        "n_sim_spikes": n_sim_spikes,
        "n_target_spikes": n_tgt_spikes,
        "pearson_r": corr,
        "model_spikes": model_spikes,
        "no_spikes": no_spikes,
        "wrong_firing_rate": wrong_firing_rate,
        "broad_spikes": broad_spikes,
        "excessive_sag": excessive_sag,
        "parameters_at_bounds": params_at_bounds,
    }


# ===========================================================================
# Main Entry Point: fit_proposal
# ===========================================================================

def fit_proposal(
    proposal: ModelProposal,
    specimen_id: int,
    data_dir: str,
    dt: float = 0.025,
    epochs: int = 300,
    lr: float = 0.02,
    max_duration_ms: float = 1200.0,
) -> DiagnosticReport:
    """
    Fit an agent-proposed model to a PV+ neuron's recordings.

    This is the function that replaces sga.py's placeholder _run_inner_loop().

    Steps:
        1. Build Jaxley cell from proposal's channel list
        2. Load training sweep from Allen data
        3. Set up trainable parameters with appropriate bounds
        4. Run gradient descent
        5. Compute diagnostics
        6. Return a DiagnosticReport for the outer loop

    Args:
        proposal:    ModelProposal from the LLM
        specimen_id: Allen Cell Types Database specimen ID
        data_dir:    Path to Allen data cache (from allen_downloader.py)
        dt:          Jaxley simulation timestep in ms
        epochs:      Number of Adam optimisation steps
        lr:          Learning rate
        max_duration_ms: Max simulation window in ms

    Returns:
        DiagnosticReport with real fitting results
    """
    data_dir = Path(data_dir)
    logger.info(
        f"  fit_proposal: channels={proposal.channels}, "
        f"specimen={specimen_id}, epochs={epochs}"
    )

    # ------------------------------------------------------------------
    # Step 1: Build cell from proposal
    # ------------------------------------------------------------------
    cell, trainable, error = build_cell_from_proposal(proposal)
    if error:
        logger.error(f"  Cell construction failed: {error}")
        return DiagnosticReport(
            proposal=proposal,
            specimen_id=specimen_id,
            final_loss=float("inf"),
            no_spikes=True,
        )

    n_params = len(trainable)
    param_names = [t["name"] for t in trainable]
    logger.info(f"  Trainable parameters ({n_params}): {param_names}")

    # ------------------------------------------------------------------
    # Step 2: Load training data
    # ------------------------------------------------------------------
    try:
        ctc = CellTypesCache(manifest_file=str(data_dir / "manifest.json"))
        with open(data_dir / "sweep_index.json") as f:
            sweep_index = json.load(f)

        sweep = load_training_sweep(ctc, specimen_id, sweep_index)
        stimulus, t_max = prepare_stimulus(sweep, dt)
        target_v = prepare_target(sweep, dt)
        target_v_jnp = jnp.array(target_v)
    except Exception as e:
        logger.error(f"  Data loading failed: {e}")
        return DiagnosticReport(
            proposal=proposal,
            specimen_id=specimen_id,
            final_loss=float("inf"),
            no_spikes=True,
        )

    # ------------------------------------------------------------------
    # Step 2b: Window to main stimulus region
    # ------------------------------------------------------------------
    stimulus, target_v_jnp, t_max = window_to_main_stimulus(
        stimulus, target_v_jnp, dt, max_duration_ms)

    logger.info(f"  Stimulus window: {len(stimulus)} steps, {t_max:.0f} ms, "
                f"stim range [{np.min(stimulus):.3f}, {np.max(stimulus):.3f}] nA")

    # ------------------------------------------------------------------
    # Step 3: Set up simulation and trainable parameters
    # ------------------------------------------------------------------
    try:
        cell = setup_simulation(cell, stimulus, dt, t_max)

        for t_info in trainable:
            cell.make_trainable(t_info["name"])

        opt_params = cell.get_parameters()

        # Build sigmoid transforms from trainable bounds
        transforms = []
        for t_info in trainable:
            transforms.append({
                t_info["name"]: SigmoidTransform(
                    lower=t_info["lower"],
                    upper=t_info["upper"],
                )
            })
        transform = ParamTransform(transforms)

    except Exception as e:
        logger.error(f"  Simulation setup failed: {e}\n{traceback.format_exc()}")
        return DiagnosticReport(
            proposal=proposal,
            specimen_id=specimen_id,
            final_loss=float("inf"),
            no_spikes=True,
        )

    # ------------------------------------------------------------------
    # Step 4: Build loss function and run gradient descent
    # ------------------------------------------------------------------
    param_names = [t["name"] for t in trainable]
    logger.info(f"  Trainable parameters ({len(param_names)}): {param_names}")
 
    loss_fn = build_generalized_loss_fn(cell, target_v_jnp, dt, transform, param_names)
 
    # --- Cosine LR schedule with linear warmup ---
    warmup_epochs = min(20, epochs // 10)
    schedule = optax.join_schedules(
        schedules=[
            optax.linear_schedule(init_value=lr * 0.1, end_value=lr,
                                  transition_steps=warmup_epochs),
            optax.cosine_decay_schedule(init_value=lr,
                                        decay_steps=epochs - warmup_epochs,
                                        alpha=0.01),  # decays to 1% of peak LR
        ],
        boundaries=[warmup_epochs],
    )
 
    optimizer = optax.chain(
        optax.clip_by_global_norm(5.0),   # tighter global norm clipping
        optax.adam(learning_rate=schedule),
    )
    opt_state = optimizer.init(opt_params)
 
    # JIT-compiled training step (no per-element clipping — handled by chain above)
    @jit
    def step(params, opt_state):
        loss_val, grads = value_and_grad(loss_fn)(params)
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss_val
 
    losses = []
    best_loss = float("inf")
    best_params = None
    nan_count = 0
    max_nan = 5
    patience = 50            # stop if no improvement for this many epochs
    epochs_since_best = 0
    divergence_threshold = 3.0  # rollback if loss > best * this factor
 
    logger.info(f"  Starting optimisation: {epochs} epochs, lr={lr} (cosine schedule)")
 
    for epoch in range(epochs):
        try:
            opt_params, opt_state, loss_val = step(opt_params, opt_state)
            loss_float = float(loss_val)
        except Exception as e:
            logger.warning(f"  Epoch {epoch}: simulation error — {e}")
            nan_count += 1
            if nan_count >= max_nan:
                logger.error(f"  {max_nan} consecutive failures, stopping")
                break
            # Rollback to best params on exception
            if best_params is not None:
                opt_params = jax.tree.map(lambda x: x.copy(), best_params)
                opt_state = optimizer.init(opt_params)
            continue
 
        if np.isnan(loss_float) or np.isinf(loss_float):
            nan_count += 1
            if nan_count >= max_nan:
                logger.warning(f"  {max_nan} consecutive NaN/inf, stopping")
                break
            # Rollback to best params on NaN
            if best_params is not None:
                opt_params = jax.tree.map(lambda x: x.copy(), best_params)
                opt_state = optimizer.init(opt_params)
            continue
        else:
            nan_count = 0
 
        losses.append(loss_float)
 
        if loss_float < best_loss:
            best_loss = loss_float
            best_params = jax.tree.map(lambda x: x.copy(), opt_params)
            epochs_since_best = 0
        else:
            epochs_since_best += 1
 
        # Rollback if loss has diverged significantly from best
        if (best_params is not None
            and best_loss > 0
            and loss_float > best_loss * divergence_threshold):
            logger.info(f"    Epoch {epoch:4d}  loss={loss_float:.4f} >> "
                        f"best={best_loss:.4f}, rolling back")
            opt_params = jax.tree.map(lambda x: x.copy(), best_params)
            opt_state = optimizer.init(opt_params)
            epochs_since_best = 0  # reset patience after rollback
 
        # Patience-based early stopping
        if epochs_since_best >= patience and epoch > warmup_epochs + patience:
            logger.info(f"    Early stopping: no improvement for {patience} epochs "
                        f"(best={best_loss:.4f})")
            break
 
        if epoch % 20 == 0 or epoch == epochs - 1:
            logger.info(f"    Epoch {epoch:4d}  loss={loss_float:.4f}  "
                        f"best={best_loss:.4f}")
 
    # ------------------------------------------------------------------
    # Step 5: Extract fitted parameters and run final simulation
    # ------------------------------------------------------------------
    # IMPORTANT: Always use best_params, not the last epoch's params
    if best_params is None:
        logger.error("  No valid parameters found during optimisation")
        return DiagnosticReport(
            proposal=proposal,
            specimen_id=specimen_id,
            final_loss=float("inf"),
            no_spikes=True,
        )
 
    # Use best_params for everything downstream
    opt_params = best_params
 
    fitted = transform.forward(best_params)
    fitted_dict = {}
    for i, name in enumerate(param_names):
        val = fitted[i][name]
        if isinstance(val, (list, np.ndarray, jnp.ndarray)):
            fitted_dict[name] = float(np.asarray(val).flatten()[0])
        else:
            fitted_dict[name] = float(val)
 
    logger.info(f"  Fitted parameters: {fitted_dict}")
    logger.info(f"  Best loss: {best_loss:.4f}")
 
    # Final simulation with best params
    try:
        param_state = None
        for i, name in enumerate(param_names):
            param_state = cell.data_set(name, fitted[i][name], param_state)
 
        v_final = jx.integrate(cell, param_state=param_state, delta_t=dt)
        v_sim = np.array(v_final[0])
    except Exception as e:
        logger.error(f"  Final simulation failed: {e}")
        return DiagnosticReport(
            proposal=proposal,
            specimen_id=specimen_id,
            final_loss=best_loss,
            no_spikes=True,
        )

    # ------------------------------------------------------------------
    # Step 6: Compute diagnostics and return report
    # ------------------------------------------------------------------
    target_np = np.array(target_v_jnp)
    diag = compute_diagnostics(
        v_sim, target_np, dt, proposal, fitted_dict, trainable, best_loss
    )

    # Store fitted params on the proposal for the heap
    proposal.fitted_params = fitted_dict
    proposal.loss = best_loss

    report = DiagnosticReport(
        proposal=proposal,
        specimen_id=specimen_id,
        final_loss=diag["final_loss"],
        n_sim_spikes=diag["n_sim_spikes"],
        n_target_spikes=diag["n_target_spikes"],
        pearson_r=diag["pearson_r"],
        model_spikes=diag["model_spikes"],
        no_spikes=diag["no_spikes"],
        wrong_firing_rate=diag["wrong_firing_rate"],
        broad_spikes=diag["broad_spikes"],
        excessive_sag=diag["excessive_sag"],
        parameters_at_bounds=diag["parameters_at_bounds"],
    )

    logger.info(f"  Result: loss={best_loss:.4f}, spikes={diag['n_sim_spikes']}/{diag['n_target_spikes']}, r={diag['pearson_r']:.3f}")

    return report


# ===========================================================================
# CLI for standalone testing
# ===========================================================================

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(
        description="Test generalized fitter with a sample proposal"
    )
    parser.add_argument("--data-dir", type=str, default="cell_types_data")
    parser.add_argument("--specimen-id", type=int, default=None,
                        help="Specimen ID (default: first valid cell)")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--channels", nargs="+",
                        default=["Na", "K", "Leak", "Kv3"],
                        help="Channels to test (default: Na K Leak Kv3)")

    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    # Find a valid specimen if not specified
    if args.specimen_id is None:
        with open(data_dir / "sweep_index.json") as f:
            sweep_index = json.load(f)
        valid = [int(sid) for sid, e in sweep_index.items() if e.get("valid")]
        if not valid:
            print("No valid cells found. Run allen_downloader.py first.")
            exit(1)
        specimen_id = valid[0]
        print(f"Using first valid cell: {specimen_id}")
    else:
        specimen_id = args.specimen_id

    # Build a test proposal
    test_proposal = ModelProposal(
        proposal_id=0,
        iteration=0,
        channels=args.channels,
        param_config={},  # will use defaults
        rationale=f"CLI test with channels: {args.channels}",
    )

    print(f"\n{'='*60}")
    print(f"Testing generalized fitter")
    print(f"  Channels: {args.channels}")
    print(f"  Specimen: {specimen_id}")
    print(f"  Epochs:   {args.epochs}")
    print(f"{'='*60}\n")

    report = fit_proposal(
        test_proposal,
        specimen_id=specimen_id,
        data_dir=str(data_dir),
        epochs=args.epochs,
    )

    print(f"\n{'='*60}")
    print("DIAGNOSTIC REPORT")
    print(f"{'='*60}")
    print(report.generate_feedback())

    print(f"\n{'='*60}")
    print("FITTED PARAMETERS")
    print(f"{'='*60}")
    for k, v in test_proposal.fitted_params.items():
        print(f"  {k}: {v:.6f}")