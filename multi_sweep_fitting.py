"""
Multi-Sweep Fitting Extension for general_fit.py
=================================================

KEY DESIGN: Each sweep gets its own pre-stimulated cell clone built at
loss-construction time. The loss function sets the same parameters on
all cells via param_state, runs jx.integrate on each independently,
and averages the per-sweep losses.

This avoids calling cell.delete_stimuli()/cell.stimulate() inside the
JIT-compiled loss function. Those are Python side effects that modify
cell.externals in place, corrupting JAX's traced computation graph and
causing NaN gradients on every call after the first JIT trace.
"""

import logging
import numpy as np
import jax
import jax.numpy as jnp
import jaxley as jx

logger = logging.getLogger(__name__)


# ===========================================================================
# Multi-Sweep Data Loading
# ===========================================================================

def load_and_prepare_sweeps(ctc, specimen_id, sweep_index, dt=0.025,
                            max_duration_ms=1200.0, n_sweeps=3):
    """
    Load and prepare multiple training sweeps for multi-sweep fitting.
    """
    from sim_fit import load_multiple_sweeps, load_training_sweep
    from sim_fit import prepare_stimulus, prepare_target

    if n_sweeps <= 1:
        sweep = load_training_sweep(ctc, specimen_id, sweep_index)
        raw_sweeps = [sweep]
    else:
        raw_sweeps = load_multiple_sweeps(ctc, specimen_id, sweep_index,
                                          n_sweeps=n_sweeps)

    from general_fit import (extract_baseline, window_to_main_stimulus,
                             _build_shared_loss_components)

    prepared = []
    for i, sweep in enumerate(raw_sweeps):
        stimulus, t_max = prepare_stimulus(sweep, dt)
        target_v = prepare_target(sweep, dt)

        baseline_v = extract_baseline(stimulus, target_v, dt)

        target_v_jnp = jnp.array(target_v)
        stimulus_w, target_w, t_max_w = window_to_main_stimulus(
            stimulus, target_v_jnp, dt, max_duration_ms)

        shared = _build_shared_loss_components(target_w, dt)

        amp = sweep.get("stimulus_amplitude", 0) or 0
        n_spk = sweep.get("num_spikes", shared.get("raw_target_spike_count", 0))

        logger.info(f"  Sweep {i+1}/{len(raw_sweeps)}: "
                    f"#{sweep['sweep_number']} ({amp:.0f} pA, "
                    f"{int(shared.get('raw_target_spike_count', 0))} spikes, "
                    f"{len(stimulus_w)} steps)")

        prepared.append({
            "stimulus": stimulus_w,
            "target_v": target_w,
            "baseline_v": baseline_v,
            "sweep_number": sweep["sweep_number"],
            "stimulus_amplitude": amp,
            "num_spikes": n_spk,
            "shared": shared,
            "t_max": t_max_w,
        })

    return prepared


# ===========================================================================
# Build Per-Sweep Cells
# ===========================================================================

def build_sweep_cells(proposal, sweep_data_list, dt, adaptive_limits=None):
    """
    Build one pre-stimulated cell per sweep for multi-sweep fitting.

    Each cell is an independent clone with the same channel composition
    and geometry, but stimulated with a different sweep's waveform.
    All cells share the same trainable parameter names.

    Returns:
        cells: list of jx.Compartment (one per sweep, each pre-stimulated
               and with trainable parameters registered)
        trainable: list of trainable param dicts (from first cell build)
        param_names: list of parameter name strings
        opt_params: initial optimizer parameters (from first cell)
    """
    from general_fit import build_cell_from_proposal
    from sim_fit import setup_simulation
    from jaxley.optimize.transforms import ParamTransform, SigmoidTransform

    cells = []
    trainable = None

    for i, sd in enumerate(sweep_data_list):
        cell, t, error = build_cell_from_proposal(proposal,
                                                   adaptive_limits=adaptive_limits)
        if error:
            raise ValueError(f"Cell construction failed for sweep {i}: {error}")

        cell = setup_simulation(cell, sd["stimulus"], dt, sd["t_max"])

        if trainable is None:
            trainable = t

        # Make parameters trainable on each cell
        for t_info in trainable:
            cell.make_trainable(t_info["name"])

        cells.append(cell)

    # Get opt_params and build transform from first cell
    opt_params = cells[0].get_parameters()
    param_names = [t["name"] for t in trainable]

    transforms = []
    for t_info in trainable:
        bound_range = t_info["upper"] - t_info["lower"]
        buffer = bound_range * 0.001
        transforms.append({
            t_info["name"]: SigmoidTransform(
                lower=t_info["lower"] - buffer,
                upper=t_info["upper"] + buffer,
            )
        })
    transform = ParamTransform(transforms)

    return cells, trainable, param_names, opt_params, transform


# ===========================================================================
# Multi-Sweep Loss Builders
# ===========================================================================

def _build_multisweep_phase1_loss_fn(cells, sweep_data_list, dt, transform,
                                      param_names, mse_weight=0.1,
                                      stats_weight=5.0,
                                      spike_count_weight=300.0):
    """
    Phase 1 multi-sweep loss: average of per-sweep (spike_count + stats + MSE).

    Each sweep has its own pre-stimulated cell. No side effects inside
    the loss function — just param_state setting and jx.integrate calls.
    """
    n_sweeps = len(sweep_data_list)
    sweep_targets = [sd["target_v"] for sd in sweep_data_list]
    sweep_shareds = [sd["shared"] for sd in sweep_data_list]

    def loss_fn(opt_params):
        params = transform.forward(opt_params)

        total_loss = jnp.array(0.0)

        for sw_idx in range(n_sweeps):
            cell = cells[sw_idx]
            target_v = sweep_targets[sw_idx]
            shared = sweep_shareds[sw_idx]

            param_state = None
            for i, name in enumerate(param_names):
                param_state = cell.data_set(name, params[i][name], param_state)

            try:
                v = jx.integrate(cell, param_state=param_state, delta_t=dt)
                v_sim = v[0]
                n = min(len(v_sim), len(target_v))
                v_s, v_t = v_sim[:n], target_v[:n]

                mse = jnp.mean((v_s - v_t) ** 2)

                n_windows = shared["n_windows"]
                win_size = shared["win_size"]
                sim_means, sim_stds = [], []
                for w in range(n_windows):
                    start = w * win_size
                    end = start + win_size if w < n_windows - 1 else n
                    win = v_s[start:end]
                    sim_means.append(jnp.mean(win))
                    sim_stds.append(jnp.std(win))
                sim_means = jnp.stack(sim_means)
                sim_stds = jnp.stack(sim_stds)
                stats_loss = (
                    jnp.mean(jnp.abs(sim_means / shared["mean_scale"]
                                      - shared["tgt_means"] / shared["mean_scale"]))
                    + jnp.mean(jnp.abs(sim_stds / shared["std_scale"]
                                        - shared["tgt_stds"] / shared["std_scale"]))
                )

                spike_k = shared["spike_k"]
                spike_threshold = shared["spike_threshold"]
                target_spike_count = shared["target_spike_count"]
                p = jax.nn.sigmoid(spike_k * (v_s - spike_threshold))
                dp = jnp.diff(p)
                soft_events = jax.nn.relu(dp)
                soft_count = jnp.sum(soft_events)
                spike_loss = (soft_count / target_spike_count - 1.0) ** 2

                sweep_loss = (mse_weight * mse
                              + stats_weight * stats_loss
                              + spike_count_weight * spike_loss)
            except Exception:
                sweep_loss = jnp.array(float("inf"))

            total_loss = total_loss + sweep_loss

        return total_loss / n_sweeps

    return loss_fn


def _build_multisweep_phase2_loss_fn(cells, sweep_data_list, dt, transform,
                                      param_names, mse_weight=0.1,
                                      stats_weight=5.0,
                                      spike_count_weight=300.0,
                                      spike_timing_weight=100.0):
    """
    Phase 2 multi-sweep loss: average of per-sweep full loss (+ spike timing).
    """
    n_sweeps = len(sweep_data_list)
    sweep_targets = [sd["target_v"] for sd in sweep_data_list]
    sweep_shareds = [sd["shared"] for sd in sweep_data_list]

    def loss_fn(opt_params):
        params = transform.forward(opt_params)

        total_loss = jnp.array(0.0)

        for sw_idx in range(n_sweeps):
            cell = cells[sw_idx]
            target_v = sweep_targets[sw_idx]
            shared = sweep_shareds[sw_idx]

            param_state = None
            for i, name in enumerate(param_names):
                param_state = cell.data_set(name, params[i][name], param_state)

            try:
                v = jx.integrate(cell, param_state=param_state, delta_t=dt)
                v_sim = v[0]
                n = min(len(v_sim), len(target_v))
                v_s, v_t = v_sim[:n], target_v[:n]

                mse = jnp.mean((v_s - v_t) ** 2)

                n_windows = shared["n_windows"]
                win_size = shared["win_size"]
                sim_means, sim_stds = [], []
                for w in range(n_windows):
                    start = w * win_size
                    end = start + win_size if w < n_windows - 1 else n
                    win = v_s[start:end]
                    sim_means.append(jnp.mean(win))
                    sim_stds.append(jnp.std(win))
                sim_means = jnp.stack(sim_means)
                sim_stds = jnp.stack(sim_stds)
                stats_loss = (
                    jnp.mean(jnp.abs(sim_means / shared["mean_scale"]
                                      - shared["tgt_means"] / shared["mean_scale"]))
                    + jnp.mean(jnp.abs(sim_stds / shared["std_scale"]
                                        - shared["tgt_stds"] / shared["std_scale"]))
                )

                spike_k = shared["spike_k"]
                spike_threshold = shared["spike_threshold"]
                target_spike_count = shared["target_spike_count"]
                p = jax.nn.sigmoid(spike_k * (v_s - spike_threshold))
                dp = jnp.diff(p)
                soft_events = jax.nn.relu(dp)
                soft_count = jnp.sum(soft_events)
                spike_loss = (soft_count / target_spike_count - 1.0) ** 2

                timing_kernel = shared["timing_kernel"]
                tgt_smoothed = shared["tgt_smoothed"]
                sim_smoothed = jnp.convolve(soft_events, timing_kernel, mode='same')
                sim_smoothed = sim_smoothed / jnp.maximum(jnp.max(sim_smoothed), 1e-8)
                n_t = min(len(sim_smoothed), len(tgt_smoothed))
                timing_loss = jnp.mean(
                    (sim_smoothed[:n_t] - tgt_smoothed[:n_t]) ** 2)

                sweep_loss = (mse_weight * mse
                              + stats_weight * stats_loss
                              + spike_count_weight * spike_loss
                              + spike_timing_weight * timing_loss)
            except Exception:
                sweep_loss = jnp.array(float("inf"))

            total_loss = total_loss + sweep_loss

        return total_loss / n_sweeps

    return loss_fn


# ===========================================================================
# Multi-Sweep Diagnostics
# ===========================================================================

def compute_multisweep_diagnostics(cell, sweep_data_list, dt, transform,
                                    param_names, fitted_params_raw,
                                    proposal, trainable, best_loss):
    """
    Run final simulation on all sweeps and aggregate diagnostics.

    Uses a fresh cell for each sweep to avoid JAX tracer leaks.
    """
    from sim_fit import setup_simulation
    from general_fit import build_cell_from_proposal

    fitted = transform.forward(fitted_params_raw)
    fitted_dict = {}
    for i, name in enumerate(param_names):
        val = fitted[i][name]
        fitted_dict[name] = (float(np.array(val).flatten()[0])
                             if hasattr(val, 'shape') else float(val))

    per_sweep = []

    for sw_data in sweep_data_list:
        try:
            fresh_cell, _, _ = build_cell_from_proposal(proposal)
            fresh_cell = setup_simulation(
                fresh_cell, sw_data["stimulus"], dt, sw_data["t_max"])

            param_state = None
            for i, name in enumerate(param_names):
                param_state = fresh_cell.data_set(name, fitted[i][name], param_state)

            v_final = jx.integrate(fresh_cell, param_state=param_state, delta_t=dt)
            v_sim = np.array(v_final[0])
        except Exception as e:
            logger.warning(f"    Diagnostic sim failed for sweep "
                           f"#{sw_data['sweep_number']}: {e}")
            v_sim = np.zeros(len(np.array(sw_data["target_v"])))

        target_np = np.array(sw_data["target_v"])
        n = min(len(v_sim), len(target_np))

        spike_threshold = -20.0
        sim_crossings = np.diff((v_sim[:n] > spike_threshold).astype(int)) > 0
        tgt_crossings = np.diff((target_np[:n] > spike_threshold).astype(int)) > 0
        n_sim = int(np.sum(sim_crossings))
        n_tgt = int(np.sum(tgt_crossings))

        corr = float(np.corrcoef(v_sim[:n], target_np[:n])[0, 1])
        if np.isnan(corr):
            corr = 0.0

        per_sweep.append({
            "sweep_number": sw_data["sweep_number"],
            "stimulus_amplitude": sw_data["stimulus_amplitude"],
            "n_sim_spikes": n_sim,
            "n_target_spikes": n_tgt,
            "pearson_r": corr,
        })

    primary = per_sweep[0]
    for ps in per_sweep:
        if ps["n_target_spikes"] > 0:
            primary = ps
            break

    total_sim = sum(ps["n_sim_spikes"] for ps in per_sweep)
    total_tgt = sum(ps["n_target_spikes"] for ps in per_sweep)
    mean_r = float(np.mean([ps["pearson_r"] for ps in per_sweep]))

    logger.info(f"  Multi-sweep diagnostics ({len(per_sweep)} sweeps):")
    for ps in per_sweep:
        logger.info(f"    Sweep #{ps['sweep_number']} "
                    f"({ps['stimulus_amplitude']:.0f} pA): "
                    f"{ps['n_sim_spikes']}/{ps['n_target_spikes']} spikes, "
                    f"r={ps['pearson_r']:.3f}")
    logger.info(f"  Aggregate: {total_sim}/{total_tgt} total spikes, "
                f"mean_r={mean_r:.3f}")

    return {
        "n_sim_spikes": primary["n_sim_spikes"],
        "n_target_spikes": primary["n_target_spikes"],
        "pearson_r": primary["pearson_r"],
        "model_spikes": primary["n_sim_spikes"] > 0,
        "no_spikes": (primary["n_target_spikes"] > 0
                      and primary["n_sim_spikes"] == 0),
        "wrong_firing_rate": (
            abs(primary["n_sim_spikes"] - primary["n_target_spikes"])
            > max(5, primary["n_target_spikes"] * 0.3)
        ) if primary["n_target_spikes"] > 0 else False,
        "broad_spikes": False,
        "excessive_sag": False,
        "per_sweep": per_sweep,
        "total_sim_spikes": total_sim,
        "total_target_spikes": total_tgt,
        "mean_pearson_r": mean_r,
    }