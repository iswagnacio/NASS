"""
Generalized Jaxley Fitter — Fit Any Agent-Proposed Model
=========================================================

Bridges sga.py's ModelProposal with Jaxley's fitting infrastructure.
Dynamically builds a Jaxley compartment from whatever channels the LLM
proposes, applies gradient safety clamping, runs gradient descent, and
returns a DiagnosticReport.

Bound architecture:
    1. LLM proposes param_config (biophysical judgment — informed by trace features)
    2. FALLBACK_PARAM_BOUNDS used for any parameter the LLM omits
    3. Gradient safety clamping (from auto_bounds) prevents NaN — the ONLY override
    4. Geometry bounds derived from LLM's proposal radius/capacitance (±25%/±30%)

The LLM sees trace features in its prompt and makes all biophysical decisions.
This module does NOT second-guess the LLM's choices — it only prevents NaN.

Training uses phased loss:
    Phase 1 (epochs 0 to phase_switch): spike_count + stats + MSE only (no timing)
    Phase 2 (epochs phase_switch to end): full loss with spike_timing added
    This prevents the optimizer from suppressing spiking to avoid timing penalties.
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

from channels import NaCortical, Kv3, IM, IAHP, IT, ICaL, IH, CHANNEL_REGISTRY
from sga import ModelProposal, DiagnosticReport
from sim_fit import (
    load_training_sweep,
    prepare_stimulus,
    prepare_target,
    setup_simulation,
)

from auto_bounds import (
    extract_trace_features,
    get_adaptive_hard_limits,
    clamp_to_gradient_safety,
)

logger = logging.getLogger(__name__)


# ===========================================================================
# Channel Resolution
# ===========================================================================

# REVERTED: NaCortical caused numerical instability (NaN every 10 epochs, 0 spikes).
# Going back to standard HH Na which achieved 44/56 spikes in pre-refactor testing.
# NaCortical available as "NaCortical" if needed for future investigation.
BUILTIN_CHANNELS = {"Na": Na, "NaCortical": NaCortical, "K": K, "Leak": Leak}
CUSTOM_CHANNELS = {name: info["class"] for name, info in CHANNEL_REGISTRY.items()}
ALL_CHANNELS = {**BUILTIN_CHANNELS, **CUSTOM_CHANNELS}

# Fallback bounds used when the LLM omits a parameter from param_config.
# These are conservative defaults — the LLM should override them with
# informed choices based on the trace features it receives in its prompt.
FALLBACK_PARAM_BOUNDS = {
    "Na_gNa":    {"init": 0.10, "lower": 0.01,  "upper": 0.20},  # Reverted to standard HH
    "K_gK":      {"init": 0.005, "lower": 0.001, "upper": 0.05},
    "Leak_gLeak":{"init": 0.0001,"lower": 1e-5,  "upper": 0.002},
    "Leak_eLeak":{"init": -70.0, "lower": -85.0, "upper": -50.0},
    "Kv3_gKv3":  {"init": 0.005, "lower": 5e-4,  "upper": 0.03},
    "IM_gM":     {"init": 7e-5,  "lower": 1e-6,  "upper": 0.005},
    "IAHP_gAHP": {"init": 1e-4,  "lower": 1e-6,  "upper": 0.005},
    "IT_gT":     {"init": 1e-4,  "lower": 1e-6,  "upper": 0.005},
    "ICaL_gCaL": {"init": 1e-4,  "lower": 1e-6,  "upper": 0.005},
    "IH_gH":     {"init": 2e-5,  "lower": 1e-6,  "upper": 0.001},
    "eNa":       {"init": 55.0,  "lower": 40.0,  "upper": 70.0},
    "eK":        {"init": -90.0, "lower": -110.0, "upper": -70.0},
    "capacitance":{"init": 1.0,  "lower": 0.5,   "upper": 2.0},
    "radius":    {"init": 10.0,  "lower": 3.0,   "upper": 12.0},
}

# Backwards-compatible alias
DEFAULT_PARAM_BOUNDS = FALLBACK_PARAM_BOUNDS

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

GLOBAL_TRAINABLE = ["eNa", "eK", "capacitance", "radius"]


# ===========================================================================
# Gradient Safety Clamping
# ===========================================================================

def _clamp_param_bounds(name: str, cfg: dict,
                        adaptive_limits: tuple = None) -> dict:
    """
    Clamp LLM-proposed bounds for gradient safety ONLY.

    If adaptive_limits are provided (from auto_bounds.get_adaptive_hard_limits),
    uses spike-count-adaptive limits. Otherwise uses static fallback.
    """
    if adaptive_limits is not None:
        hard_ceilings, hard_floors, hard_globals = adaptive_limits
        return clamp_to_gradient_safety(
            name, cfg, hard_ceilings, hard_floors, hard_globals
        )

    # Static fallback (only used if auto_bounds unavailable)
    cfg = dict(cfg)
    _STATIC_CEILINGS = {
        "Na_gNa": 0.50, "K_gK": 0.10, "Leak_gLeak": 0.005,
        "Kv3_gKv3": 0.10, "IM_gM": 0.01, "IAHP_gAHP": 0.01,
        "IT_gT": 0.01, "ICaL_gCaL": 0.01, "IH_gH": 0.005,
    }
    _STATIC_FLOORS = {"eNa": 30.0, "eK": -120.0, "capacitance": 0.2, "radius": 1.5}
    _STATIC_GLOBALS = {"eNa": 115.0, "eK": -55.0, "capacitance": 3.0, "radius": 20.0}

    if name in _STATIC_CEILINGS and cfg.get("upper", 0) > _STATIC_CEILINGS[name]:
        cfg["upper"] = _STATIC_CEILINGS[name]
    if name in _STATIC_FLOORS and cfg.get("lower", 0) < _STATIC_FLOORS[name]:
        cfg["lower"] = _STATIC_FLOORS[name]
    if name in _STATIC_GLOBALS and cfg.get("upper", 0) > _STATIC_GLOBALS[name]:
        cfg["upper"] = _STATIC_GLOBALS[name]

    if "init" in cfg:
        cfg["init"] = max(cfg["init"], cfg.get("lower", cfg["init"]))
        cfg["init"] = min(cfg["init"], cfg.get("upper", cfg["init"]))

    return cfg


def _ensure_init_margin(init_val, lower, upper, margin_frac=0.05):
    """Ensure init is not at sigmoid boundary (prevents logit → ±inf)."""
    margin = (upper - lower) * margin_frac
    return float(max(lower + margin, min(upper - margin, init_val)))


# ===========================================================================
# Geometry Bounds from LLM Proposal
# ===========================================================================

def _geometry_bounds_from_proposal(proposal: ModelProposal) -> dict:
    """
    Derive radius/capacitance bounds from the LLM's proposal geometry.

    The LLM proposes explicit geometry values (radius, capacitance).
    We use them as init and constrain the optimizer to ±25% for radius
    and ±30% for capacitance. This prevents the optimizer from inflating
    radius to suppress spiking (the radius-inflation exploit), while
    still letting it fine-tune geometry within a reasonable range.

    Radius is tighter (±25%) because the optimizer strongly exploits
    radius inflation to dilute current density and avoid spiking.
    Capacitance is slightly looser (±30%) as it's less exploitable.

    This is NOT hardcoding — each neuron type gets different geometry via
    the LLM's per-cell proposal. PV+ FS cells get small radii (~8 µm),
    pyramidals get larger (~15 µm), etc.

    Returns dict of {param_name: {init, lower, upper}} overrides.
    """
    overrides = {}

    if hasattr(proposal, 'radius') and proposal.radius is not None:
        r = float(proposal.radius)
        overrides["radius"] = {
            "init": r,
            "lower": max(r * 0.75, 1.5),   # at least 1.5 µm
            "upper": r * 1.25,
        }

    if hasattr(proposal, 'capacitance') and proposal.capacitance is not None:
        c = float(proposal.capacitance)
        overrides["capacitance"] = {
            "init": c,
            "lower": max(c * 0.7, 0.2),
            "upper": c * 1.3,
        }

    return overrides


# ===========================================================================
# Build Cell from Proposal
# ===========================================================================

def build_cell_from_proposal(proposal: ModelProposal,
                              adaptive_limits: tuple = None) -> tuple:
    """
    Build a Jaxley single-compartment cell from a ModelProposal.

    The LLM's param_config is used directly — only gradient safety
    clamping is applied. For parameters the LLM omits, FALLBACK_PARAM_BOUNDS
    provides defaults. Geometry bounds are derived from the proposal's
    radius/capacitance fields (±25%/±30%).

    Args:
        proposal:         ModelProposal from the LLM
        adaptive_limits:  Gradient safety limits from get_adaptive_hard_limits()

    Returns:
        (cell, trainable_params, error_message)
    """
    invalid = [ch for ch in proposal.channels if ch not in ALL_CHANNELS]
    if invalid:
        return None, None, f"Unknown channels: {invalid}. Available: {list(ALL_CHANNELS.keys())}"

    channels = list(proposal.channels)
    for required in ["Na", "K", "Leak"]:
        if required not in channels:
            channels.insert(0, required)
            logger.info(f"  Auto-added required channel: {required}")

    try:
        comp = jx.Compartment()
        for ch_name in channels:
            comp.insert(ALL_CHANNELS[ch_name]())

        comp.set("radius", proposal.radius)
        comp.set("length", proposal.length)
        comp.set("capacitance", proposal.capacitance)
        comp.set("axial_resistivity", 100.0)
        comp.set("eNa", 50.0)
        comp.set("eK", -90.0)

        # Set initial values from LLM's param_config or fallback
        for ch_name in channels:
            for param_name in CHANNEL_CONDUCTANCE_PARAMS.get(ch_name, []):
                if param_name in proposal.param_config:
                    init_val = _clamp_param_bounds(
                        param_name, proposal.param_config[param_name],
                        adaptive_limits
                    ).get("init")
                elif param_name in FALLBACK_PARAM_BOUNDS:
                    cfg = FALLBACK_PARAM_BOUNDS[param_name]
                    init_val = _ensure_init_margin(cfg["init"], cfg["lower"], cfg["upper"])
                else:
                    continue
                try:
                    comp.set(param_name, init_val)
                except Exception as e:
                    logger.warning(f"  Could not set {param_name}={init_val}: {e}")

    except Exception as e:
        return None, None, f"Cell construction failed: {e}\n{traceback.format_exc()}"

    # Collect trainable parameters
    trainable = []
    geometry_overrides = _geometry_bounds_from_proposal(proposal)

    for ch_name in channels:
        for param_name in CHANNEL_CONDUCTANCE_PARAMS.get(ch_name, []):
            if param_name in proposal.param_config:
                cfg = _clamp_param_bounds(param_name, proposal.param_config[param_name],
                                          adaptive_limits)
            elif param_name in FALLBACK_PARAM_BOUNDS:
                cfg = FALLBACK_PARAM_BOUNDS[param_name]
            else:
                continue
            trainable.append({"name": param_name, "lower": cfg["lower"], "upper": cfg["upper"]})

    for param_name in GLOBAL_TRAINABLE:
        # Priority: param_config > geometry_overrides > fallback
        if param_name in proposal.param_config:
            cfg = _clamp_param_bounds(param_name, proposal.param_config[param_name],
                                      adaptive_limits)
        elif param_name in geometry_overrides:
            cfg = _clamp_param_bounds(param_name, geometry_overrides[param_name],
                                      adaptive_limits)
        elif param_name in FALLBACK_PARAM_BOUNDS:
            cfg = FALLBACK_PARAM_BOUNDS[param_name]
        else:
            continue
        trainable.append({"name": param_name, "lower": cfg["lower"], "upper": cfg["upper"]})

    return comp, trainable, None


# ===========================================================================
# Build Loss Functions (Phased)
# ===========================================================================

def _build_shared_loss_components(target_v, dt, n_windows=10,
                                   spike_threshold=-20.0):
    """
    Pre-compute all target-derived quantities shared by Phase 1 and Phase 2
    loss functions. Avoids duplicating expensive target preprocessing.

    Returns a dict of JAX arrays and scalars.
    """
    target_np = np.array(target_v)
    n_total = len(target_np)
    win_size = n_total // n_windows

    tgt_crossings = np.diff((target_np > spike_threshold).astype(int)) > 0
    target_spike_count = max(float(np.sum(tgt_crossings)), 1.0)

    target_means, target_stds = [], []
    for w in range(n_windows):
        start = w * win_size
        end = start + win_size if w < n_windows - 1 else n_total
        window = target_np[start:end]
        target_means.append(float(np.mean(window)))
        target_stds.append(float(np.std(window)))

    spike_k = 0.5
    sigma_ms = 2.0
    sigma_steps = sigma_ms / dt
    half_kernel = int(3.0 * sigma_steps)
    kernel_x = jnp.arange(-half_kernel, half_kernel + 1, dtype=jnp.float32)
    timing_kernel = jnp.exp(-kernel_x ** 2 / (2.0 * sigma_steps ** 2))
    timing_kernel = timing_kernel / jnp.sum(timing_kernel)

    tgt_p = 1.0 / (1.0 + np.exp(-spike_k * (target_np - spike_threshold)))
    tgt_dp = np.diff(tgt_p)
    tgt_events = np.maximum(tgt_dp, 0.0)
    tgt_smoothed_np = np.convolve(tgt_events, np.array(timing_kernel), mode='same')
    tgt_peak = float(np.max(tgt_smoothed_np)) if float(np.max(tgt_smoothed_np)) > 0 else 1.0
    tgt_smoothed = jnp.array(tgt_smoothed_np / tgt_peak, dtype=jnp.float32)

    return {
        "n_total": n_total,
        "n_windows": n_windows,
        "win_size": win_size,
        "target_spike_count": target_spike_count,
        "tgt_means": jnp.array(target_means),
        "tgt_stds": jnp.array(target_stds),
        "mean_scale": 8.0,
        "std_scale": 4.0,
        "spike_k": spike_k,
        "spike_threshold": spike_threshold,
        "timing_kernel": timing_kernel,
        "tgt_smoothed": tgt_smoothed,
        "tgt_peak": tgt_peak,
        "sigma_ms": sigma_ms,
        "sigma_steps": sigma_steps,
    }


def _build_phase1_loss_fn(cell, target_v, dt, transform, param_names,
                           shared, mse_weight=0.1, stats_weight=5.0,
                           spike_count_weight=300.0):
    """
    Phase 1 loss: spike_count + windowed_stats + MSE.
    NO spike timing penalty.

    This lets the optimizer find spiking solutions without being penalized
    for spike misalignment. Once spiking is established, Phase 2 adds
    timing to refine spike positions.
    """
    n_windows = shared["n_windows"]
    win_size = shared["win_size"]
    target_spike_count = shared["target_spike_count"]
    tgt_means = shared["tgt_means"]
    tgt_stds = shared["tgt_stds"]
    mean_scale = shared["mean_scale"]
    std_scale = shared["std_scale"]
    spike_k = shared["spike_k"]
    spike_threshold = shared["spike_threshold"]

    def loss_fn(opt_params):
        params = transform.forward(opt_params)
        param_state = None
        for i, name in enumerate(param_names):
            param_state = cell.data_set(name, params[i][name], param_state)

        try:
            v = jx.integrate(cell, param_state=param_state, delta_t=dt)
            v_sim = v[0]
            n = min(len(v_sim), len(target_v))
            v_s, v_t = v_sim[:n], target_v[:n]

            mse = jnp.mean((v_s - v_t) ** 2)

            sim_means, sim_stds = [], []
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

            p = jax.nn.sigmoid(spike_k * (v_s - spike_threshold))
            dp = jnp.diff(p)
            soft_events = jax.nn.relu(dp)
            soft_count = jnp.sum(soft_events)
            spike_loss = (soft_count / target_spike_count - 1.0) ** 2

            # NO timing_loss in Phase 1
            total = (mse_weight * mse + stats_weight * stats_loss
                     + spike_count_weight * spike_loss)
        except Exception:
            total = jnp.array(float("inf"))
        return total

    return loss_fn


def _build_phase2_loss_fn(cell, target_v, dt, transform, param_names,
                           shared, mse_weight=0.1, stats_weight=5.0,
                           spike_count_weight=300.0,
                           spike_timing_weight=100.0):
    """
    Phase 2 (full) loss: spike_count + spike_timing + windowed_stats + MSE.

    All four components active. Used after Phase 1 has established spiking.
    """
    n_windows = shared["n_windows"]
    win_size = shared["win_size"]
    target_spike_count = shared["target_spike_count"]
    tgt_means = shared["tgt_means"]
    tgt_stds = shared["tgt_stds"]
    mean_scale = shared["mean_scale"]
    std_scale = shared["std_scale"]
    spike_k = shared["spike_k"]
    spike_threshold = shared["spike_threshold"]
    timing_kernel = shared["timing_kernel"]
    tgt_smoothed = shared["tgt_smoothed"]
    tgt_peak = shared["tgt_peak"]

    def loss_fn(opt_params):
        params = transform.forward(opt_params)
        param_state = None
        for i, name in enumerate(param_names):
            param_state = cell.data_set(name, params[i][name], param_state)

        try:
            v = jx.integrate(cell, param_state=param_state, delta_t=dt)
            v_sim = v[0]
            n = min(len(v_sim), len(target_v))
            v_s, v_t = v_sim[:n], target_v[:n]

            mse = jnp.mean((v_s - v_t) ** 2)

            sim_means, sim_stds = [], []
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

            p = jax.nn.sigmoid(spike_k * (v_s - spike_threshold))
            dp = jnp.diff(p)
            soft_events = jax.nn.relu(dp)
            soft_count = jnp.sum(soft_events)
            spike_loss = (soft_count / target_spike_count - 1.0) ** 2

            sim_smoothed = jnp.convolve(soft_events, timing_kernel, mode='same') / tgt_peak
            n_timing = min(len(sim_smoothed), len(tgt_smoothed))
            timing_loss = jnp.mean((sim_smoothed[:n_timing] - tgt_smoothed[:n_timing]) ** 2)

            total = (mse_weight * mse + stats_weight * stats_loss
                     + spike_count_weight * spike_loss + spike_timing_weight * timing_loss)
        except Exception:
            total = jnp.array(float("inf"))
        return total

    return loss_fn


def build_generalized_loss_fn(cell, target_v, dt, transform, param_names,
                              n_windows=10, mse_weight=0.1,
                              stats_weight=5.0,
                              spike_count_weight=300.0,
                              spike_timing_weight=100.0,
                              spike_threshold=-20.0):
    """
    Backward-compatible wrapper: builds full (Phase 2) loss function.

    Code that calls build_generalized_loss_fn directly (e.g. sim_fit.py)
    gets the same behavior as before. The phased training only applies
    inside fit_proposal.
    """
    shared = _build_shared_loss_components(target_v, dt, n_windows, spike_threshold)
    return _build_phase2_loss_fn(cell, target_v, dt, transform, param_names,
                                 shared, mse_weight, stats_weight,
                                 spike_count_weight, spike_timing_weight)


# ===========================================================================
# Stimulus Windowing
# ===========================================================================

def window_to_main_stimulus(stimulus, target_v_jnp, dt, max_duration_ms=1200.0):
    """Window stimulus and target to the main current step."""
    stim_np = np.array(stimulus)
    stim_threshold = np.max(np.abs(stim_np)) * 0.1
    above = np.abs(stim_np) > stim_threshold
    active_indices = np.where(above)[0]

    if len(active_indices) > 0:
        breaks = np.where(np.diff(active_indices) > int(5.0 / dt))[0]
        main_block = max(np.split(active_indices, breaks + 1), key=len) if len(breaks) > 0 else active_indices

        pre_pad, post_pad = int(50.0 / dt), int(100.0 / dt)
        start_idx = max(0, main_block[0] - pre_pad)
        end_idx = min(len(stim_np), main_block[-1] + post_pad)
        max_samples = int(max_duration_ms / dt) + 1
        if (end_idx - start_idx) > max_samples:
            end_idx = start_idx + max_samples

        stimulus = stimulus[start_idx:end_idx]
        target_v_jnp = target_v_jnp[start_idx:end_idx]
        t_max = len(stimulus) * dt
        logger.info(f"  Windowed to main stimulus: [{start_idx*dt:.0f}–{end_idx*dt:.0f}] ms, "
                    f"{len(stimulus)} steps, {t_max:.0f} ms")
    elif len(stimulus) * dt > max_duration_ms:
        n_keep = int(max_duration_ms / dt) + 1
        stimulus = stimulus[:n_keep]
        target_v_jnp = target_v_jnp[:n_keep]
        t_max = max_duration_ms
    else:
        t_max = len(stimulus) * dt

    return stimulus, target_v_jnp, t_max


# ===========================================================================
# Compute Diagnostics
# ===========================================================================

def compute_diagnostics(v_sim, target_v, dt, proposal, fitted_params, trainable, final_loss):
    """Compute all diagnostic flags for the outer loop."""
    n = min(len(v_sim), len(target_v))
    v_sim, target_v = v_sim[:n], target_v[:n]
    spike_threshold = -20.0

    sim_crossings = np.diff((v_sim > spike_threshold).astype(int)) > 0
    tgt_crossings = np.diff((target_v > spike_threshold).astype(int)) > 0
    n_sim_spikes = int(np.sum(sim_crossings))
    n_tgt_spikes = int(np.sum(tgt_crossings))

    corr = float(np.corrcoef(v_sim, target_v)[0, 1]) if len(v_sim) > 1 else 0.0
    if np.isnan(corr): corr = 0.0

    no_spikes = (n_tgt_spikes > 0) and (n_sim_spikes == 0)
    wrong_firing_rate = False
    if n_tgt_spikes > 0 and n_sim_spikes > 0:
        rate_ratio = n_sim_spikes / n_tgt_spikes
        wrong_firing_rate = rate_ratio < 0.5 or rate_ratio > 2.0

    broad_spikes = False
    if n_sim_spikes >= 3:
        sim_crossing_idxs = np.where(sim_crossings)[0]
        for idx in sim_crossing_idxs[:5]:
            if idx + int(2.0 / dt) < len(v_sim):
                if np.all(v_sim[idx:idx + int(2.0 / dt)] > spike_threshold):
                    broad_spikes = True
                    break

    excessive_sag = False
    sub_mask = target_v < -70.0
    if sub_mask.sum() > 10:
        tgt_hyp = target_v[sub_mask]
        sim_hyp = v_sim[sub_mask] if sub_mask.sum() <= len(v_sim) else np.array([])
        if len(sim_hyp) > 0 and np.min(sim_hyp) < np.min(tgt_hyp) - 10.0:
            excessive_sag = True

    params_at_bounds = []
    for t_info in trainable:
        name, lower, upper = t_info["name"], t_info["lower"], t_info["upper"]
        if name in fitted_params:
            val = fitted_params[name]
            margin = (upper - lower) * 0.02
            if abs(val - lower) < margin:
                params_at_bounds.append(f"{name} (at lower={lower})")
            elif abs(val - upper) < margin:
                params_at_bounds.append(f"{name} (at upper={upper})")

    return {
        "final_loss": final_loss, "n_sim_spikes": n_sim_spikes,
        "n_target_spikes": n_tgt_spikes, "pearson_r": corr,
        "model_spikes": n_sim_spikes > 0, "no_spikes": no_spikes,
        "wrong_firing_rate": wrong_firing_rate, "broad_spikes": broad_spikes,
        "excessive_sag": excessive_sag, "parameters_at_bounds": params_at_bounds,
    }


# ===========================================================================
# Baseline Extraction Helper (used by OuterLoop too)
# ===========================================================================

def extract_baseline(stimulus_full, target_v_full, dt):
    """
    Extract pre-stimulus baseline voltage from the full (unwindowed) trace.
    Returns baseline_v array (voltage before stimulus onset).
    """
    stim_np = np.array(stimulus_full)
    target_np = np.array(target_v_full)
    stim_threshold = np.max(np.abs(stim_np)) * 0.1
    active_indices = np.where(np.abs(stim_np) > stim_threshold)[0]

    if len(active_indices) > 0:
        stim_onset_idx = active_indices[0]
        n_baseline = max(10, stim_onset_idx)
        baseline_v = target_np[:n_baseline]
    else:
        n_baseline = max(10, len(target_np) // 10)
        baseline_v = target_np[:n_baseline]

    return baseline_v


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
    phase_fraction: float = 0.5,
) -> DiagnosticReport:
    """
    Fit an agent-proposed model to a neuron's recordings.

    Uses phased loss training:
        Phase 1 (epochs 0 to phase_switch):
            spike_count + stats + MSE only. No timing penalty.
            Goal: find parameters that produce spikes.
        Phase 2 (epochs phase_switch to end):
            Full loss with spike_timing added.
            Goal: align spike timing to target.

    The phase_switch epoch is epochs * phase_fraction (default 0.5 = halfway).

    Steps:
        1. Load training data
        2. Extract baseline & window to stimulus
        3. Compute gradient safety limits from trace
        4. Build cell (LLM's param_config + geometry bounds + gradient safety)
        5. Build phased loss functions
        6. Run gradient descent with phase switching
        7. Return DiagnosticReport
    """
    data_dir = Path(data_dir)
    logger.info(f"  fit_proposal: channels={proposal.channels}, "
                f"specimen={specimen_id}, epochs={epochs}")

    # Step 1: Load training data
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
        return DiagnosticReport(proposal=proposal, specimen_id=specimen_id,
                                final_loss=float("inf"), no_spikes=True)

    # Step 2: Extract baseline, then window
    baseline_v = extract_baseline(stimulus, target_v, dt)
    logger.info(f"  Baseline: {len(baseline_v)} samples, Vrest={np.mean(baseline_v):.1f} mV")

    stimulus, target_v_jnp, t_max = window_to_main_stimulus(
        stimulus, target_v_jnp, dt, max_duration_ms)
    logger.info(f"  Stimulus window: {len(stimulus)} steps, {t_max:.0f} ms, "
                f"stim range [{np.min(stimulus):.3f}, {np.max(stimulus):.3f}] nA")

    # Step 3: Compute gradient safety limits from trace features
    target_np = np.array(target_v_jnp)
    try:
        features = extract_trace_features(target_np, dt, baseline_v=baseline_v)
        adaptive_limits = get_adaptive_hard_limits(features)
    except Exception as e:
        logger.warning(f"  Feature extraction failed ({e}), using static safety limits")
        adaptive_limits = None

    # Step 4: Build cell — LLM's param_config + geometry bounds + gradient safety
    cell, trainable, error = build_cell_from_proposal(proposal, adaptive_limits=adaptive_limits)
    if error:
        logger.error(f"  Cell construction failed: {error}")
        return DiagnosticReport(proposal=proposal, specimen_id=specimen_id,
                                final_loss=float("inf"), no_spikes=True)

    n_params = len(trainable)
    param_names = [t["name"] for t in trainable]
    logger.info(f"  Trainable parameters ({n_params}): {param_names}")

    # Step 5: Set up simulation
    try:
        cell = setup_simulation(cell, stimulus, dt, t_max)
        for t_info in trainable:
            cell.make_trainable(t_info["name"])
        opt_params = cell.get_parameters()

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
    except Exception as e:
        logger.error(f"  Simulation setup failed: {e}\n{traceback.format_exc()}")
        return DiagnosticReport(proposal=proposal, specimen_id=specimen_id,
                                final_loss=float("inf"), no_spikes=True)

    # Step 6: Build phased loss functions
    param_names = [t["name"] for t in trainable]
    logger.info(f"  Trainable parameters ({len(param_names)}): {param_names}")

    phase_switch = int(epochs * phase_fraction)
    logger.info(f"  Phased training: Phase 1 epochs [0, {phase_switch}), "
                f"Phase 2 epochs [{phase_switch}, {epochs})")

    shared = _build_shared_loss_components(target_v_jnp, dt)

    logger.info(f"  Loss function: target_spikes={int(shared['target_spike_count'])}, "
                f"mse_weight=0.1, stats_weight=5.0, "
                f"spike_count_weight=300.0, spike_timing_weight=100.0")
    logger.info(f"  Spike timing kernel: sigma={shared['sigma_ms']}ms "
                f"({shared['sigma_steps']:.0f} steps), "
                f"kernel length={len(shared['timing_kernel'])}")

    loss_fn_p1 = _build_phase1_loss_fn(cell, target_v_jnp, dt, transform,
                                        param_names, shared)
    loss_fn_p2 = _build_phase2_loss_fn(cell, target_v_jnp, dt, transform,
                                        param_names, shared)

    logger.info(f"  Phase 1 loss: spike_count=300, stats=5, MSE=0.1, "
                f"spike_timing=0 (disabled)")
    logger.info(f"  Phase 2 loss: spike_count=300, spike_timing=100, "
                f"stats=5, MSE=0.1")

    # Step 7: Set up optimizer
    n_extra_channels = max(0, len(proposal.channels) - 3)
    stiffness_factor = 1.0 / (1.0 + 0.5 * n_extra_channels)
    lr_effective = lr * stiffness_factor
    logger.info(f"  LR scaling: base={lr}, stiffness_factor={stiffness_factor:.2f} "
                f"({n_extra_channels} extra channels), effective={lr_effective:.4f}")

    clip_norm = 5.0 * np.sqrt(8.0 / max(len(param_names), 1))
    clip_norm = max(clip_norm, 1.0)
    logger.info(f"  Gradient clip norm: {clip_norm:.2f} (for {len(param_names)} params)")

    warmup_epochs = min(30, epochs // 5)
    schedule = optax.join_schedules(
        schedules=[
            optax.linear_schedule(init_value=lr_effective * 0.01,
                                  end_value=lr_effective,
                                  transition_steps=warmup_epochs),
            optax.cosine_decay_schedule(init_value=lr_effective,
                                        decay_steps=epochs - warmup_epochs,
                                        alpha=0.01),
        ],
        boundaries=[warmup_epochs],
    )
    optimizer = optax.chain(optax.clip_by_global_norm(clip_norm), optax.adam(learning_rate=schedule))
    opt_state = optimizer.init(opt_params)

    # Step 8: JIT-compiled training steps for BOTH phases
    @jit
    def step_phase1(params, opt_state):
        loss_val, grads = value_and_grad(loss_fn_p1)(params)
        grad_finite = jax.tree.reduce(
            lambda a, b: a & b,
            jax.tree.map(lambda g: jnp.all(jnp.isfinite(g)), grads),
        )
        safe_grads = jax.tree.map(
            lambda g: jnp.where(grad_finite, g, jnp.zeros_like(g)), grads)
        updates, new_opt_state = optimizer.update(safe_grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss_val, grad_finite

    @jit
    def step_phase2(params, opt_state):
        loss_val, grads = value_and_grad(loss_fn_p2)(params)
        grad_finite = jax.tree.reduce(
            lambda a, b: a & b,
            jax.tree.map(lambda g: jnp.all(jnp.isfinite(g)), grads),
        )
        safe_grads = jax.tree.map(
            lambda g: jnp.where(grad_finite, g, jnp.zeros_like(g)), grads)
        updates, new_opt_state = optimizer.update(safe_grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss_val, grad_finite

    # Step 9: Training loop with phase switching
    losses = []
    best_loss, best_params = float("inf"), None
    nan_count, max_nan, nan_grad_count = 0, 15, 0
    patience, epochs_since_best = 50, 0
    divergence_threshold = 3.0
    jitter_scale = 0.01
    rng = np.random.RandomState(42)
    current_phase = 1

    logger.info(f"  Starting optimisation: {epochs} epochs, lr={lr_effective:.4f} "
                f"(cosine schedule, warmup={warmup_epochs})")

    for epoch in range(epochs):
        # ---- Phase switching ----
        if epoch == phase_switch and current_phase == 1:
            current_phase = 2
            logger.info(f"  === PHASE SWITCH at epoch {epoch}: "
                        f"adding spike_timing_weight=100.0 ===")
            # Reset best-loss tracking since the loss function changed.
            # Phase 2 loss includes timing so absolute values will be higher.
            # Keep best_params as the starting point — don't lose Phase 1 progress.
            best_loss = float("inf")
            epochs_since_best = 0

        # ---- Select step function for current phase ----
        step_fn = step_phase1 if current_phase == 1 else step_phase2

        try:
            opt_params, opt_state, loss_val, grad_ok = step_fn(opt_params, opt_state)
            loss_float = float(loss_val)
            grad_was_finite = bool(grad_ok)
        except Exception:
            nan_count += 1
            if nan_count >= max_nan: break
            if best_params is not None:
                opt_params = jax.tree.map(
                    lambda x: x + jitter_scale * jax.numpy.array(rng.randn(*x.shape).astype(np.float32)),
                    best_params)
                opt_state = optimizer.init(opt_params)
            continue

        if np.isnan(loss_float) or np.isinf(loss_float):
            nan_count += 1
            if nan_count >= max_nan: break
            if best_params is not None:
                opt_params = jax.tree.map(
                    lambda x: x + jitter_scale * jax.numpy.array(rng.randn(*x.shape).astype(np.float32)),
                    best_params)
                opt_state = optimizer.init(opt_params)
            if nan_count > 3: jitter_scale = min(jitter_scale * 1.5, 0.1)
            continue
        else:
            nan_count = 0
            jitter_scale = 0.01

        if not grad_was_finite:
            nan_grad_count += 1
            if nan_grad_count % 10 == 0:
                logger.info(f"    Epoch {epoch}: NaN gradient (zeroed), total={nan_grad_count}")
            continue

        losses.append(loss_float)
        if loss_float < best_loss:
            best_loss = loss_float
            best_params = jax.tree.map(lambda x: x.copy(), opt_params)
            epochs_since_best = 0
        else:
            epochs_since_best += 1

        if best_params is not None and best_loss > 0 and loss_float > best_loss * divergence_threshold:
            opt_params = jax.tree.map(lambda x: x.copy(), best_params)
            opt_state = optimizer.init(opt_params)
            epochs_since_best = 0

        if epochs_since_best >= patience and epoch > warmup_epochs + patience:
            logger.info(f"    Early stopping: no improvement for {patience} epochs (best={best_loss:.4f})")
            break

        if epoch % 20 == 0 or epoch == epochs - 1:
            phase_tag = f"[P{current_phase}]"
            logger.info(f"    Epoch {epoch:4d}  loss={loss_float:.4f}  best={best_loss:.4f}  {phase_tag}")

    # Step 10: Extract fitted parameters
    if best_params is None:
        logger.error("  No valid parameters found during optimisation")
        return DiagnosticReport(proposal=proposal, specimen_id=specimen_id,
                                final_loss=float("inf"), no_spikes=True)

    fitted = transform.forward(best_params)
    fitted_dict = {}
    for i, name in enumerate(param_names):
        val = fitted[i][name]
        fitted_dict[name] = float(np.asarray(val).flatten()[0]) if isinstance(val, (list, np.ndarray, jnp.ndarray)) else float(val)

    logger.info(f"  Fitted parameters: {fitted_dict}")
    logger.info(f"  Best loss: {best_loss:.4f}")

    try:
        param_state = None
        for i, name in enumerate(param_names):
            param_state = cell.data_set(name, fitted[i][name], param_state)
        v_final = jx.integrate(cell, param_state=param_state, delta_t=dt)
        v_sim = np.array(v_final[0])
    except Exception as e:
        logger.error(f"  Final simulation failed: {e}")
        return DiagnosticReport(proposal=proposal, specimen_id=specimen_id,
                                final_loss=best_loss, no_spikes=True)

    # Step 11: Diagnostics
    target_np = np.array(target_v_jnp)
    diag = compute_diagnostics(v_sim, target_np, dt, proposal, fitted_dict, trainable, best_loss)

    proposal.fitted_params = fitted_dict
    proposal.loss = best_loss

    report = DiagnosticReport(
        proposal=proposal, specimen_id=specimen_id,
        final_loss=diag["final_loss"], n_sim_spikes=diag["n_sim_spikes"],
        n_target_spikes=diag["n_target_spikes"], pearson_r=diag["pearson_r"],
        model_spikes=diag["model_spikes"], no_spikes=diag["no_spikes"],
        wrong_firing_rate=diag["wrong_firing_rate"], broad_spikes=diag["broad_spikes"],
        excessive_sag=diag["excessive_sag"], parameters_at_bounds=diag["parameters_at_bounds"],
    )
    logger.info(f"  Result: loss={best_loss:.4f}, spikes={diag['n_sim_spikes']}/{diag['n_target_spikes']}, r={diag['pearson_r']:.3f}")
    return report


# ===========================================================================
# CLI
# ===========================================================================

if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="cell_types_data")
    parser.add_argument("--specimen-id", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--channels", nargs="+", default=["Na", "K", "Leak", "Kv3"])
    parser.add_argument("--phase-fraction", type=float, default=0.5,
                        help="Fraction of epochs for Phase 1 (no timing loss). Default 0.5")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if args.specimen_id is None:
        with open(data_dir / "sweep_index.json") as f:
            sweep_index = json.load(f)
        valid = [int(sid) for sid, e in sweep_index.items() if e.get("valid")]
        specimen_id = valid[0] if valid else None
    else:
        specimen_id = args.specimen_id

    test_proposal = ModelProposal(proposal_id=0, iteration=0, channels=args.channels,
                                   param_config={}, rationale=f"CLI test: {args.channels}")
    report = fit_proposal(test_proposal, specimen_id=specimen_id,
                          data_dir=str(data_dir), epochs=args.epochs,
                          phase_fraction=args.phase_fraction)
    print(report.generate_feedback())