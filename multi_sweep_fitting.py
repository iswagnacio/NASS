"""
Multi-Sweep Fitting Module for NASS
====================================

Provides the multi-sweep loss functions and data loading that general_fit.py
imports when n_sweeps > 1.

Architecture:
    - All sweeps are padded/windowed to the SAME length (required for JIT)
    - Each sweep gets its own target trace, spike count, and loss components
    - Per-sweep data_stimuli are PRE-BUILT outside the loss function (eager mode)
      and captured as closure constants — this avoids calling cell.data_stimulate()
      inside JIT-traced code, which causes NaN gradients due to cell side effects
    - Per-sweep losses are summed with optional weighting by spike count
      (sweeps with more spikes contribute more to the total loss)

Key Jaxley API:
    cell.stimulate(current)        — static, NOT JIT-compatible
    cell.data_stimulate(current, data_stimuli)  — builds data_stimuli dict (has cell side effects)
    jx.integrate(cell, param_state=..., data_stimuli=...)  — JIT/grad-compatible

CRITICAL: cell.data_stimulate() modifies internal cell state (externals indexing).
Calling it inside a JIT-traced function causes accumulated side effects that
corrupt the computation graph, producing NaN gradients on backward pass.
The fix: call data_stimulate() ONCE per sweep during setup (eager mode),
then reference the pre-built data_stimuli inside the loss function.
"""

import logging
import numpy as np
import jax
import jax.numpy as jnp

import jaxley as jx

from sim_fit import (
    load_multiple_sweeps,
    prepare_stimulus,
    prepare_target,
)

from general_fit import (
    _build_shared_loss_components,
    extract_baseline,
    window_to_main_stimulus,
)

logger = logging.getLogger(__name__)


# ===========================================================================
# Data Loading & Preparation
# ===========================================================================

def load_and_prepare_sweeps(ctc, specimen_id, sweep_index, dt=0.025,
                            max_duration_ms=1200.0, n_sweeps=3):
    """
    Load multiple sweeps and prepare them for multi-sweep fitting.

    All sweeps are windowed to the stimulus epoch and then padded/truncated
    to the SAME length (the maximum across sweeps). This is required because
    JAX JIT recompiles for different array shapes.

    Returns a list of dicts, each containing:
        - stimulus: jnp array (n_steps,)
        - target_v: jnp array (n_steps,)
        - baseline_v: np array
        - t_max: float
        - sweep_number: int
        - stimulus_amplitude: float
        - shared: dict from _build_shared_loss_components()
    """
    raw_sweeps = load_multiple_sweeps(ctc, specimen_id, sweep_index,
                                      n_sweeps=n_sweeps)

    # Prepare each sweep: resample, extract baseline, window
    prepared = []
    for sweep in raw_sweeps:
        stim, t_max = prepare_stimulus(sweep, dt)
        target_v = prepare_target(sweep, dt)

        # Extract baseline from pre-stimulus period of FULL trace
        baseline_v = extract_baseline(stim, target_v, dt)

        # Window to stimulus epoch
        stim_win, target_win, t_max_win = window_to_main_stimulus(
            stim, jnp.array(target_v), dt, max_duration_ms)

        prepared.append({
            "stimulus": np.array(stim_win),
            "target_v_np": np.array(target_win),
            "baseline_v": baseline_v,
            "t_max": t_max_win,
            "sweep_number": sweep["sweep_number"],
            "stimulus_amplitude": sweep.get("stimulus_amplitude", 0.0),
            "num_spikes": sweep.get("num_spikes", 0),
        })

    # Pad all sweeps to the same length (max across sweeps)
    max_len = max(len(p["stimulus"]) for p in prepared)
    logger.info(f"  Multi-sweep: padding all sweeps to {max_len} steps "
                f"({max_len * dt:.0f} ms)")

    result = []
    for p in prepared:
        n = len(p["stimulus"])
        if n < max_len:
            # Pad stimulus with zeros (no current), target with last value
            stim_padded = np.zeros(max_len, dtype=np.float64)
            stim_padded[:n] = p["stimulus"]

            target_padded = np.full(max_len, p["target_v_np"][-1],
                                    dtype=np.float64)
            target_padded[:n] = p["target_v_np"]
        else:
            stim_padded = p["stimulus"][:max_len]
            target_padded = p["target_v_np"][:max_len]

        target_v_jnp = jnp.array(target_padded)

        # Build shared loss components for this sweep
        shared = _build_shared_loss_components(target_v_jnp, dt)

        result.append({
            "stimulus": jnp.array(stim_padded),
            "target_v": target_v_jnp,
            "baseline_v": p["baseline_v"],
            "t_max": max_len * dt,
            "sweep_number": p["sweep_number"],
            "stimulus_amplitude": p["stimulus_amplitude"],
            "num_spikes": p.get("num_spikes", 0),
            "shared": shared,
            "original_len": n,
        })

    return result


# ===========================================================================
# Pre-build data_stimuli (EAGER mode — outside JIT)
# ===========================================================================

def prebuild_data_stimuli(cell, sweep_data_list):
    """
    Pre-build Jaxley data_stimuli dicts for each sweep in EAGER mode.

    CRITICAL: cell.data_stimulate() has side effects on cell internals
    (e.g. appending to external stimulus indexing structures). Calling it
    inside a JIT-traced loss function causes these side effects to accumulate
    during tracing, corrupting the computation graph and producing NaN
    gradients on the backward pass.

    By calling data_stimulate() here (once per sweep, eagerly), we get
    pure JAX data structures (dicts of arrays) that can be safely captured
    in the loss function closure and used under JIT/grad.

    NOTE: Each call registers one external on the cell. For N sweeps, the
    cell will have N external registrations. This is expected — each sweep's
    data_stimuli points to its own registered slot. Do NOT call
    cell.data_stimulate() again later (e.g. in _count_probe_spikes) or
    additional registrations will accumulate.

    Returns a list of data_stimuli dicts, one per sweep.
    """
    result = []
    for si, sd in enumerate(sweep_data_list):
        stim = sd["stimulus"]
        data_stimuli = None
        data_stimuli = cell.data_stimulate(stim, data_stimuli)
        result.append(data_stimuli)

    logger.info(f"  Pre-built data_stimuli for {len(result)} sweeps "
                f"(eager mode, outside JIT)")
    return result


# ===========================================================================
# Multi-Sweep Loss Functions
# ===========================================================================

def _single_sweep_phase1_loss(v_sim, target_v, shared,
                               mse_weight=0.1, stats_weight=5.0,
                               spike_count_weight=300.0,
                               baseline_weight=50.0):
    """
    Compute Phase 1 loss for a single sweep (no timing).
    Pure JAX — no side effects, fully differentiable.
    """
    # Use Python min(), NOT jnp.minimum() — len() returns static ints,
    # but jnp.minimum returns a traced JAX scalar that can't be used
    # as a slice index under JIT (causes IndexError on v_sim[:n]).
    n = min(len(v_sim), len(target_v))
    v_s = v_sim[:n]
    v_t = target_v[:n]

    # MSE
    mse = jnp.mean((v_s - v_t) ** 2)

    # Windowed stats
    n_windows = shared["n_windows"]
    win_size = shared["win_size"]
    tgt_means = shared["tgt_means"]
    tgt_stds = shared["tgt_stds"]
    mean_scale = shared["mean_scale"]
    std_scale = shared["std_scale"]

    sim_means = []
    sim_stds = []
    for w in range(n_windows):
        start = w * win_size
        end = start + win_size if w < n_windows - 1 else n
        win = v_s[start:end]
        sim_means.append(jnp.mean(win))
        sim_stds.append(jnp.std(win))
    sim_means = jnp.stack(sim_means)
    sim_stds = jnp.stack(sim_stds)

    stats_loss = (jnp.mean(jnp.abs(sim_means / mean_scale - tgt_means / mean_scale))
                  + jnp.mean(jnp.abs(sim_stds / std_scale - tgt_stds / std_scale)))

    # Soft spike count
    # For target > 0: use ratio-based error (scale-invariant for comparing
    # across sweeps with different spike counts).
    # For target = 0 (subthreshold sweep): use absolute count penalty
    # (any spike is bad, but don't blow up the loss as soft_count grows).
    spike_k = shared["spike_k"]
    spike_threshold = shared["spike_threshold"]
    target_spike_count = shared["target_spike_count"]  # floored to >= 1
    raw_target = shared.get("raw_target_spike_count", 0)

    p = jax.nn.sigmoid(spike_k * (v_s - spike_threshold))
    dp = jnp.diff(p)
    soft_events = jax.nn.relu(dp)
    soft_count = jnp.sum(soft_events)

    if raw_target > 0:
        # Normal case: ratio-based relative error
        spike_loss = (soft_count / target_spike_count - 1.0) ** 2
    else:
        # Subthreshold case (target=0): absolute count penalty, saturating.
        # Use tanh to bound the penalty so a wildly over-firing model
        # doesn't dominate the total loss. Penalty saturates near 1.0
        # as soft_count grows beyond ~10.
        spike_loss = jnp.tanh(soft_count / 5.0) ** 2

    # Baseline Vrest loss
    n_baseline = shared["n_baseline_est"]
    baseline_loss = jnp.mean((v_s[:n_baseline] - v_t[:n_baseline]) ** 2)

    return (mse_weight * mse + stats_weight * stats_loss
            + spike_count_weight * spike_loss
            + baseline_weight * baseline_loss)


def _single_sweep_phase2_loss(v_sim, target_v, shared,
                               mse_weight=0.1, stats_weight=5.0,
                               spike_count_weight=300.0,
                               spike_timing_weight=100.0):
    """
    Compute Phase 2 loss for a single sweep (with timing).
    """
    # Phase 1 components
    p1 = _single_sweep_phase1_loss(v_sim, target_v, shared,
                                    mse_weight, stats_weight,
                                    spike_count_weight)

    # Timing loss — use Python min() for static slice index (same as phase1)
    n = min(len(v_sim), len(target_v))
    v_s = v_sim[:n]
    spike_k = shared["spike_k"]
    spike_threshold = shared["spike_threshold"]
    timing_kernel = shared["timing_kernel"]
    tgt_smoothed = shared["tgt_smoothed"]

    p = jax.nn.sigmoid(spike_k * (v_s - spike_threshold))
    dp = jnp.diff(p)
    soft_events = jax.nn.relu(dp)

    sim_smoothed = jnp.convolve(soft_events, timing_kernel, mode='same')
    sim_smoothed = sim_smoothed / jnp.maximum(jnp.max(sim_smoothed), 1e-8)
    n_t = min(len(sim_smoothed), len(tgt_smoothed))
    timing_loss = jnp.mean((sim_smoothed[:n_t] - tgt_smoothed[:n_t]) ** 2)

    return p1 + spike_timing_weight * timing_loss


def _build_multisweep_phase1_loss_fn(cell, sweep_data_list, dt,
                                      transform, param_names,
                                      mse_weight=0.1, stats_weight=5.0,
                                      spike_count_weight=300.0,
                                      pre_data_stimuli=None):
    """
    Build a Phase 1 multi-sweep loss function.

    The loss is the weighted sum of per-sweep losses. Sweeps with more
    target spikes get proportionally higher weight (they carry more
    information about the model's firing behavior).

    CRITICAL: pre_data_stimuli must be provided — these are the data_stimuli
    dicts pre-built by prebuild_data_stimuli() in eager mode. The loss
    function references them as closure constants, avoiding any
    cell.data_stimulate() calls inside JIT-traced code.
    """
    # Pre-compute per-sweep weights based on spike count
    spike_counts = [sd["shared"]["raw_target_spike_count"] for sd in sweep_data_list]
    total_spikes = max(sum(spike_counts), 1)
    # Weight = spike_count / total, with a floor of 0.1 for sub-threshold sweeps
    weights = [max(sc / total_spikes, 0.1) for sc in spike_counts]
    weight_sum = sum(weights)
    weights = [w / weight_sum for w in weights]  # normalize to sum=1

    # Pre-extract JAX arrays for each sweep
    targets = [sd["target_v"] for sd in sweep_data_list]
    shareds = [sd["shared"] for sd in sweep_data_list]

    n_sweeps = len(sweep_data_list)

    # Pre-build data_stimuli if not provided (backward compat, but warn)
    if pre_data_stimuli is None:
        logger.warning("  pre_data_stimuli not provided — building inside "
                       "factory (may cause NaN gradients under JIT)")
        pre_data_stimuli = []
        for sd in sweep_data_list:
            ds = None
            ds = cell.data_stimulate(sd["stimulus"], ds)
            pre_data_stimuli.append(ds)

    for i, sd in enumerate(sweep_data_list):
        tc = int(sd["shared"].get("raw_target_spike_count", 0))
        logger.info(f"    Sweep #{sd['sweep_number']}: weight={weights[i]:.2f}, "
                    f"target_spikes={tc}")

    def loss_fn(opt_params):
        # Set parameters once
        params = transform.forward(opt_params)
        param_state = None
        for i, name in enumerate(param_names):
            param_state = cell.data_set(name, params[i][name], param_state)

        total_loss = jnp.array(0.0)

        for sw_idx in range(n_sweeps):
            # Use PRE-BUILT data_stimuli — no cell.data_stimulate() here!
            # This is the critical fix: data_stimulate() has cell side effects
            # that corrupt the computation graph under JIT tracing.
            v = jx.integrate(cell, param_state=param_state,
                             data_stimuli=pre_data_stimuli[sw_idx],
                             delta_t=dt)
            v_sim = v[0]

            # Compute per-sweep loss
            sw_loss = _single_sweep_phase1_loss(
                v_sim, targets[sw_idx], shareds[sw_idx],
                mse_weight, stats_weight, spike_count_weight)

            total_loss = total_loss + weights[sw_idx] * sw_loss

        return total_loss

    return loss_fn


def _build_multisweep_phase2_loss_fn(cell, sweep_data_list, dt,
                                      transform, param_names,
                                      mse_weight=0.1, stats_weight=5.0,
                                      spike_count_weight=300.0,
                                      spike_timing_weight=100.0,
                                      pre_data_stimuli=None):
    """
    Build a Phase 2 multi-sweep loss function (with timing).

    CRITICAL: pre_data_stimuli must be provided — see Phase 1 docstring.
    """
    spike_counts = [sd["shared"]["raw_target_spike_count"] for sd in sweep_data_list]
    total_spikes = max(sum(spike_counts), 1)
    weights = [max(sc / total_spikes, 0.1) for sc in spike_counts]
    weight_sum = sum(weights)
    weights = [w / weight_sum for w in weights]

    targets = [sd["target_v"] for sd in sweep_data_list]
    shareds = [sd["shared"] for sd in sweep_data_list]

    n_sweeps = len(sweep_data_list)

    # Pre-build data_stimuli if not provided (backward compat, but warn)
    if pre_data_stimuli is None:
        logger.warning("  pre_data_stimuli not provided — building inside "
                       "factory (may cause NaN gradients under JIT)")
        pre_data_stimuli = []
        for sd in sweep_data_list:
            ds = None
            ds = cell.data_stimulate(sd["stimulus"], ds)
            pre_data_stimuli.append(ds)

    def loss_fn(opt_params):
        params = transform.forward(opt_params)
        param_state = None
        for i, name in enumerate(param_names):
            param_state = cell.data_set(name, params[i][name], param_state)

        total_loss = jnp.array(0.0)

        for sw_idx in range(n_sweeps):
            # Use PRE-BUILT data_stimuli — no cell.data_stimulate() here!
            v = jx.integrate(cell, param_state=param_state,
                             data_stimuli=pre_data_stimuli[sw_idx],
                             delta_t=dt)
            v_sim = v[0]

            sw_loss = _single_sweep_phase2_loss(
                v_sim, targets[sw_idx], shareds[sw_idx],
                mse_weight, stats_weight, spike_count_weight,
                spike_timing_weight)

            total_loss = total_loss + weights[sw_idx] * sw_loss

        return total_loss

    return loss_fn


# ===========================================================================
# Multi-Sweep Diagnostics
# ===========================================================================

def compute_multisweep_diagnostics(cell, sweep_data_list, dt,
                                    transform, param_names,
                                    best_params, proposal, trainable,
                                    best_loss):
    """
    Run final simulation on each sweep and aggregate diagnostics.

    Returns a dict compatible with general_fit.compute_diagnostics() output.
    Reports primary sweep (most spikes) stats for backward compatibility,
    plus per-sweep breakdown.

    NOTE: This runs in eager mode (not JIT), so data_stimulate() is safe here.
    """
    fitted = transform.forward(best_params)

    # Set params once
    param_state = None
    for i, name in enumerate(param_names):
        param_state = cell.data_set(name, fitted[i][name], param_state)

    per_sweep = []
    spike_threshold = -20.0

    for sd in sweep_data_list:
        try:
            data_stimuli = None
            data_stimuli = cell.data_stimulate(sd["stimulus"], data_stimuli)
            v = jx.integrate(cell, param_state=param_state,
                             data_stimuli=data_stimuli, delta_t=dt)
            v_sim = np.array(v[0])

            target_np = np.array(sd["target_v"])
            n = min(len(v_sim), len(target_np))
            v_sim = v_sim[:n]
            target_np = target_np[:n]

            sim_crossings = np.diff((v_sim > spike_threshold).astype(int)) > 0
            tgt_crossings = np.diff((target_np > spike_threshold).astype(int)) > 0
            n_sim = int(np.sum(sim_crossings))
            n_tgt = int(np.sum(tgt_crossings))

            corr = np.corrcoef(v_sim, target_np)[0, 1]
            if np.isnan(corr):
                corr = 0.0

            per_sweep.append({
                "sweep_number": sd["sweep_number"],
                "stimulus_amplitude": sd["stimulus_amplitude"],
                "n_sim_spikes": n_sim,
                "n_target_spikes": n_tgt,
                "pearson_r": float(corr),
            })
        except Exception as e:
            logger.warning(f"  Diagnostic sim failed for sweep "
                           f"#{sd['sweep_number']}: {e}")
            per_sweep.append({
                "sweep_number": sd["sweep_number"],
                "stimulus_amplitude": sd.get("stimulus_amplitude", 0),
                "n_sim_spikes": 0,
                "n_target_spikes": sd["shared"].get("raw_target_spike_count", 0),
                "pearson_r": 0.0,
            })

    # Log per-sweep results
    for ps in per_sweep:
        logger.info(f"    Sweep #{ps['sweep_number']} "
                    f"({ps['stimulus_amplitude']:.0f} pA): "
                    f"{ps['n_sim_spikes']}/{ps['n_target_spikes']} spikes, "
                    f"r={ps['pearson_r']:.3f}")

    # Aggregate: use the primary sweep (most target spikes) for top-level stats
    primary = max(per_sweep, key=lambda x: x["n_target_spikes"])

    total_sim = sum(ps["n_sim_spikes"] for ps in per_sweep)
    total_tgt = sum(ps["n_target_spikes"] for ps in per_sweep)

    return {
        "final_loss": best_loss,
        "n_sim_spikes": total_sim,
        "n_target_spikes": total_tgt,
        "pearson_r": primary["pearson_r"],
        "model_spikes": total_sim > 0,
        "no_spikes": total_sim == 0,
        "wrong_firing_rate": abs(total_sim - total_tgt) > max(5, total_tgt * 0.3),
        "broad_spikes": False,  # Could compute per-sweep if needed
        "excessive_sag": False,
        "parameters_at_bounds": [],  # Filled by caller
        "per_sweep": per_sweep,
    }