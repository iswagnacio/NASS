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

Training uses multi-start probing + phased loss:
    Probe phase: K diverse inits × 40 epochs (Phase 1 only) → pick winner
    Phase 1 (remaining epochs to phase_switch): spike_count + stats + MSE only
    Phase 2 (phase_switch to end): full loss with spike_timing added

This eliminates initialization sensitivity — the main failure mode where
50%+ of iterations land in the non-spiking basin due to unlucky init values.
"""

# ---- JAX config must come before any JAX imports ----
from jax import config
config.update("jax_enable_x64", True)

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

# IMPORTANT: "Na" maps to NaCortical (cortical fast-recovery Na channel),
# NOT to Jaxley's built-in squid-axon HH Na. NaCortical uses name="Na"
# so all param names (Na_gNa, Na_m, Na_h) are identical — drop-in replacement.
# This is required because standard HH Na inactivation recovery (τ_h ≈ 5-10 ms)
# cannot support >30 Hz firing regardless of parameter tuning.
# Cortical Nav1.1/Nav1.6 kinetics (τ_h ≈ 1-2 ms at rest) are essential for
# PV+ fast-spiking interneurons firing at 50+ Hz.
BUILTIN_CHANNELS = {"Na": NaCortical, "NaCortical": NaCortical, "K": K, "Leak": Leak}
CUSTOM_CHANNELS = {name: info["class"] for name, info in CHANNEL_REGISTRY.items()}
ALL_CHANNELS = {**BUILTIN_CHANNELS, **CUSTOM_CHANNELS}

FALLBACK_PARAM_BOUNDS = {
    "Na_gNa":    {"init": 0.10, "lower": 0.01,  "upper": 0.20},
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

DEFAULT_PARAM_BOUNDS = FALLBACK_PARAM_BOUNDS

CHANNEL_CONDUCTANCE_PARAMS = {
    "Na":          ["Na_gNa"],
    "NaCortical":  ["Na_gNa"],   # Same params as "Na" (name="Na" in class)
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
    """
    if adaptive_limits is not None:
        hard_ceilings, hard_floors, hard_globals = adaptive_limits
        return clamp_to_gradient_safety(
            name, cfg, hard_ceilings, hard_floors, hard_globals
        )

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
    ±25% for radius, ±30% for capacitance.
    """
    overrides = {}

    if hasattr(proposal, 'radius') and proposal.radius is not None:
        r = float(proposal.radius)
        overrides["radius"] = {
            "init": r,
            "lower": max(r * 0.75, 1.5),
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
    """Pre-compute target-derived quantities shared by Phase 1 and Phase 2."""
    target_np = np.array(target_v)
    n_total = len(target_np)
    win_size = n_total // n_windows

    tgt_crossings = np.diff((target_np > spike_threshold).astype(int)) > 0
    raw_target_spike_count = int(np.sum(tgt_crossings))
    # Normalizer for the scale-invariant absolute-error spike loss:
    #   spike_loss = ((soft_count - raw_target) / normalizer) ** 2
    # The floor of 3.0 prevents division explosion for target=0 (subthreshold)
    # while still penalizing spurious spikes more firmly than the old tanh branch.
    # Replaces the previous ratio-based formula (which was scale-dependent and
    # required a separate tanh branch for the target=0 case).
    spike_loss_normalizer = max(float(raw_target_spike_count), 3.0)

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
    kernel_x = jnp.arange(-half_kernel, half_kernel + 1)
    timing_kernel = jnp.exp(-kernel_x ** 2 / (2.0 * sigma_steps ** 2))
    timing_kernel = timing_kernel / jnp.sum(timing_kernel)

    tgt_p = 1.0 / (1.0 + np.exp(-spike_k * (target_np - spike_threshold)))
    tgt_dp = np.diff(tgt_p)
    tgt_events = np.maximum(tgt_dp, 0.0)
    tgt_smoothed_np = np.convolve(tgt_events, np.array(timing_kernel), mode='same')
    tgt_peak = float(np.max(tgt_smoothed_np)) if float(np.max(tgt_smoothed_np)) > 0 else 1.0
    tgt_smoothed = jnp.array(tgt_smoothed_np / tgt_peak)

    # Pre-stimulus baseline for Vrest matching. Now that the windowed trace
    # includes a pre-stimulus period, we know the first N samples are baseline
    # (zero stimulus). This gives the optimizer a clean gradient signal for eLeak.
    # We identify baseline as samples where |stimulus| < threshold (the pre-stim
    # portion we included via pre_stim_ms in window_to_main_stimulus).
    stim_check = np.array(target_v)  # target_v is a jnp array
    # Find where stimulus is essentially zero — those are the baseline samples.
    # We use the target trace's first 10% as a fallback estimate.
    n_baseline_est = max(10, n_total // 20)
    baseline_vrest = float(np.mean(target_np[:n_baseline_est]))

    return {
        "n_total": n_total,
        "n_windows": n_windows,
        "win_size": win_size,
        "raw_target_spike_count": raw_target_spike_count,
        "spike_loss_normalizer": spike_loss_normalizer,
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
        "baseline_vrest": baseline_vrest,
        "n_baseline_est": n_baseline_est,
    }


def _build_phase1_loss_fn(cell, target_v, dt, transform, param_names,
                           shared, mse_weight=0.1, stats_weight=5.0,
                           spike_count_weight=300.0,
                           baseline_weight=50.0):
    """Phase 1 loss: spike_count + windowed_stats + MSE + baseline_vrest. NO timing."""
    n_windows = shared["n_windows"]
    win_size = shared["win_size"]
    raw_target_spike_count = shared["raw_target_spike_count"]
    spike_loss_normalizer = shared["spike_loss_normalizer"]
    tgt_means = shared["tgt_means"]
    tgt_stds = shared["tgt_stds"]
    mean_scale = shared["mean_scale"]
    std_scale = shared["std_scale"]
    spike_k = shared["spike_k"]
    spike_threshold = shared["spike_threshold"]
    baseline_vrest = shared["baseline_vrest"]
    n_baseline = shared["n_baseline_est"]

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
            # Scale-invariant absolute-error spike loss with a floor-of-3
            # normalizer. Naturally bounded for target=0 (no tanh branch needed);
            # monotone around target>=1 (penalizes spurious spikes correctly).
            spike_loss = ((soft_count - raw_target_spike_count)
                          / spike_loss_normalizer) ** 2

            # Baseline Vrest loss: MSE of the pre-stimulus baseline period.
            # This directly penalizes the model's resting potential being
            # wrong, giving eLeak a strong gradient signal. Weight=50 means
            # a 12 mV offset contributes 50*(12^2)/n_baseline ~ 50*144/2000
            # ~ 3.6 per sample, which is comparable to spike count loss.
            sim_baseline = v_s[:n_baseline]
            tgt_baseline = v_t[:n_baseline]
            baseline_loss = jnp.mean((sim_baseline - tgt_baseline) ** 2)

            total = (mse_weight * mse + stats_weight * stats_loss
                     + spike_count_weight * spike_loss
                     + baseline_weight * baseline_loss)
        except Exception:
            total = jnp.array(float("inf"))
        return total

    return loss_fn


def _build_phase2_loss_fn(cell, target_v, dt, transform, param_names,
                           shared, mse_weight=0.1, stats_weight=5.0,
                           spike_count_weight=300.0,
                           spike_timing_weight=100.0,
                           baseline_weight=50.0):
    """Phase 2 (full) loss: spike_count + spike_timing + windowed_stats + MSE + baseline."""
    n_windows = shared["n_windows"]
    win_size = shared["win_size"]
    raw_target_spike_count = shared["raw_target_spike_count"]
    spike_loss_normalizer = shared["spike_loss_normalizer"]
    tgt_means = shared["tgt_means"]
    tgt_stds = shared["tgt_stds"]
    mean_scale = shared["mean_scale"]
    std_scale = shared["std_scale"]
    spike_k = shared["spike_k"]
    spike_threshold = shared["spike_threshold"]
    timing_kernel = shared["timing_kernel"]
    tgt_smoothed = shared["tgt_smoothed"]
    tgt_peak = shared["tgt_peak"]
    baseline_vrest = shared["baseline_vrest"]
    n_baseline = shared["n_baseline_est"]

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
            # Scale-invariant absolute-error spike loss (see Phase 1 note).
            spike_loss = ((soft_count - raw_target_spike_count)
                          / spike_loss_normalizer) ** 2

            # Timing loss
            sim_smoothed = jnp.convolve(soft_events, timing_kernel, mode='same')
            sim_smoothed = sim_smoothed / jnp.maximum(jnp.max(sim_smoothed), 1e-8)
            n_t = min(len(sim_smoothed), len(tgt_smoothed))
            timing_loss = jnp.mean((sim_smoothed[:n_t] - tgt_smoothed[:n_t]) ** 2)

            # Baseline Vrest loss (same as Phase 1)
            sim_baseline = v_s[:n_baseline]
            tgt_baseline = v_t[:n_baseline]
            baseline_loss = jnp.mean((sim_baseline - tgt_baseline) ** 2)

            total = (mse_weight * mse + stats_weight * stats_loss
                     + spike_count_weight * spike_loss
                     + spike_timing_weight * timing_loss
                     + baseline_weight * baseline_loss)
        except Exception:
            total = jnp.array(float("inf"))
        return total

    return loss_fn


# ===========================================================================
# Multi-Start Probe System (NEW)
# ===========================================================================

def _generate_diverse_inits(proposal, trainable, n_starts=5, rng_seed=42,
                            warm_start_params=None):
    """
    Generate n_starts diverse parameter initializations for multi-start probing.

    Strategy: Perturb the LLM's init values along the dimensions that matter
    most for basin selection (Na_gNa, Kv3_gKv3, radius).

    If warm_start_params is provided (fitted params from a previous spiking
    proposal), it replaces the LLM's init as Start 0. This guarantees at
    least one probe starts from a known-spiking configuration.

    Start 0: warm_start (if available) or LLM's original proposal
    Start 1: More excitable (higher Na_gNa, lower Kv3)
    Start 2: Less excitable (lower Na_gNa, higher Kv3)
    Start 3: Compact geometry (smaller radius + slight Na boost)
    Start 4: Random perturbation within bounds

    Returns list of (label, overrides_dict) tuples.
    """
    rng = np.random.RandomState(rng_seed)

    # Extract the LLM's proposed init values
    base_inits = {}
    for t in trainable:
        name = t["name"]
        if name in proposal.param_config and "init" in proposal.param_config[name]:
            base_inits[name] = proposal.param_config[name]["init"]

    def _get_bounds(name):
        return next((t for t in trainable if t["name"] == name), None)

    # Start 0: warm-start from previous best (if available) or LLM's original
    if warm_start_params:
        # Filter to only params that exist in this proposal's trainable set
        trainable_names = {t["name"] for t in trainable}
        ws_overrides = {k: v for k, v in warm_start_params.items()
                        if k in trainable_names}
        if ws_overrides:
            starts = [("warm_start", ws_overrides)]
            # Also include the LLM's original as a separate start
            starts.append(("original", {}))
        else:
            starts = [("original", {})]
    else:
        starts = [("original", {})]

    if n_starts >= 2:
        # Start 1: More excitable
        overrides = {}
        if "Na_gNa" in base_inits:
            b = _get_bounds("Na_gNa")
            if b:
                overrides["Na_gNa"] = base_inits["Na_gNa"] + 0.3 * (b["upper"] - base_inits["Na_gNa"])
        if "Kv3_gKv3" in base_inits:
            b = _get_bounds("Kv3_gKv3")
            if b:
                overrides["Kv3_gKv3"] = base_inits["Kv3_gKv3"] - 0.3 * (base_inits["Kv3_gKv3"] - b["lower"])
        starts.append(("excitable", overrides))

    if n_starts >= 3:
        # Start 2: Less excitable
        overrides = {}
        if "Na_gNa" in base_inits:
            b = _get_bounds("Na_gNa")
            if b:
                overrides["Na_gNa"] = base_inits["Na_gNa"] - 0.2 * (base_inits["Na_gNa"] - b["lower"])
        if "Kv3_gKv3" in base_inits:
            b = _get_bounds("Kv3_gKv3")
            if b:
                overrides["Kv3_gKv3"] = base_inits["Kv3_gKv3"] + 0.3 * (b["upper"] - base_inits["Kv3_gKv3"])
        starts.append(("inhibited", overrides))

    if n_starts >= 4:
        # Start 3: Compact geometry
        overrides = {}
        b = _get_bounds("radius")
        if b:
            r_init = base_inits.get("radius", proposal.radius)
            overrides["radius"] = r_init - 0.4 * (r_init - b["lower"])
        if "Na_gNa" in base_inits:
            b_na = _get_bounds("Na_gNa")
            if b_na:
                overrides["Na_gNa"] = base_inits["Na_gNa"] + 0.15 * (b_na["upper"] - base_inits["Na_gNa"])
        starts.append(("compact", overrides))

    if n_starts >= 5:
        # Start 4: Random within bounds
        overrides = {}
        for name in ["Na_gNa", "Kv3_gKv3", "K_gK", "eNa", "radius", "capacitance"]:
            if name in base_inits:
                b = _get_bounds(name)
                if b:
                    margin = 0.1 * (b["upper"] - b["lower"])
                    overrides[name] = rng.uniform(b["lower"] + margin, b["upper"] - margin)
        starts.append(("random", overrides))

    # Extra random starts
    for i in range(5, n_starts):
        overrides = {}
        for name in ["Na_gNa", "Kv3_gKv3", "K_gK", "eNa", "radius"]:
            if name in base_inits:
                b = _get_bounds(name)
                if b:
                    margin = 0.1 * (b["upper"] - b["lower"])
                    overrides[name] = rng.uniform(b["lower"] + margin, b["upper"] - margin)
        starts.append((f"random_{i}", overrides))

    return starts[:n_starts]


def _apply_init_overrides(opt_params, transform, trainable, overrides):
    """
    Create new opt_params with init overrides applied in optimizer space.

    Works by: forward-transform → override real values → inverse-transform.
    """
    if not overrides:
        return jax.tree.map(lambda x: x.copy(), opt_params)

    real_params = transform.forward(opt_params)
    param_names = [t["name"] for t in trainable]

    new_real = []
    for i, name in enumerate(param_names):
        if name in overrides:
            val = overrides[name]
            # Clamp to bounds with margin
            margin = (trainable[i]["upper"] - trainable[i]["lower"]) * 0.05
            val = max(trainable[i]["lower"] + margin,
                      min(trainable[i]["upper"] - margin, val))
            new_real.append({name: jnp.array(val, dtype=real_params[i][name].dtype)})
        else:
            new_real.append({name: real_params[i][name]})

    return transform.inverse(new_real)


def _run_probe(opt_params, optimizer, step_fn, n_probe_epochs):
    """
    Run a short optimization probe. Returns (best_loss, best_params, n_finite).

    CRITICAL: Saves PRE-update params (not post-update) when gradient is finite.
    The gradient is computed at the pre-update params, so finiteness is only
    guaranteed there. Saving post-update params can land in a NaN-gradient
    basin, causing main training to get 100% NaN gradients and stall.
    """
    opt_state = optimizer.init(opt_params)
    best_loss = float("inf")
    best_params = None
    n_nan = 0
    n_finite = 0
    n_exception = 0
    n_inf_loss = 0
    n_nan_grad = 0
    first_exception = None

    for epoch in range(n_probe_epochs):
        # Save PRE-update params — gradient finiteness is guaranteed here
        pre_update_params = jax.tree.map(lambda x: x.copy(), opt_params)

        try:
            opt_params, opt_state, loss_val, grad_ok = step_fn(opt_params, opt_state)
            loss_float = float(loss_val)
            grad_finite = bool(grad_ok)
        except Exception as e:
            n_nan += 1
            n_exception += 1
            if first_exception is None:
                first_exception = f"{type(e).__name__}: {str(e)[:200]}"
            if n_nan > 5:
                break
            continue

        if np.isnan(loss_float) or np.isinf(loss_float):
            n_nan += 1
            n_inf_loss += 1
            if n_nan > 5:
                break
            continue

        if not grad_finite:
            n_nan_grad += 1
            continue

        n_finite += 1
        if loss_float < best_loss:
            best_loss = loss_float
            # Save PRE-update params: the gradient was finite at THESE params,
            # so main training can start with a guaranteed-finite first step.
            best_params = pre_update_params

    if n_finite == 0 and (n_exception > 0 or n_inf_loss > 0 or n_nan_grad > 0):
        logger.warning(f"      Probe: 0/{min(epoch + 1, n_probe_epochs)} finite — "
                       f"exc={n_exception}, inf={n_inf_loss}, nan_grad={n_nan_grad}"
                       f"{f', first_exc: {first_exception}' if first_exception else ''}")

    return best_loss, best_params, n_finite


# ===========================================================================
# Diagnostic Computation
# ===========================================================================

def compute_diagnostics(v_sim, target_np, dt, proposal, fitted_dict,
                        trainable, best_loss):
    """Compute post-training diagnostics."""
    n = min(len(v_sim), len(target_np))
    v_sim = v_sim[:n]
    target_np = target_np[:n]

    spike_threshold = -20.0
    sim_crossings = np.diff((v_sim > spike_threshold).astype(int)) > 0
    tgt_crossings = np.diff((target_np > spike_threshold).astype(int)) > 0
    n_sim_spikes = int(np.sum(sim_crossings))
    n_target_spikes = int(np.sum(tgt_crossings))

    corr = np.corrcoef(v_sim, target_np)[0, 1]
    if np.isnan(corr):
        corr = 0.0

    model_spikes = n_sim_spikes > 0
    no_spikes = not model_spikes
    wrong_fr = abs(n_sim_spikes - n_target_spikes) > max(5, n_target_spikes * 0.3)

    # Check for broad spikes
    broad_spikes = False
    if model_spikes:
        sim_spike_indices = np.where(sim_crossings)[0]
        if len(sim_spike_indices) > 0:
            widths = []
            for idx in sim_spike_indices[:5]:
                above = v_sim[idx:min(idx + 200, n)] > spike_threshold
                widths.append(np.sum(above) * dt)
            avg_width = np.mean(widths)
            broad_spikes = avg_width > 3.0

    # Parameters at bounds
    at_bounds = []
    for t in trainable:
        name = t["name"]
        if name in fitted_dict:
            val = fitted_dict[name]
            rng = t["upper"] - t["lower"]
            if rng > 0:
                if (val - t["lower"]) / rng < 0.02:
                    at_bounds.append(f"{name}=lower")
                elif (t["upper"] - val) / rng < 0.02:
                    at_bounds.append(f"{name}=upper")

    # Subthreshold R2 and Vrest mismatch
    sub_thresh = -30.0
    sub_mask_tgt = target_np < sub_thresh
    sub_mask_sim = v_sim < sub_thresh
    combined_sub = sub_mask_tgt & sub_mask_sim

    if np.sum(combined_sub) > 20:
        t_sub = target_np[combined_sub]
        s_sub = v_sim[combined_sub]
        ss_res = np.sum((t_sub - s_sub) ** 2)
        ss_tot = np.sum((t_sub - np.mean(t_sub)) ** 2)
        sub_r2 = max(0.0, 1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    else:
        sub_r2 = 0.0

    if np.sum(sub_mask_tgt) > 10:
        vrest_target = float(np.mean(target_np[sub_mask_tgt]))
    else:
        vrest_target = float(np.mean(target_np[:max(10, n // 10)]))
    if np.sum(sub_mask_sim) > 10:
        vrest_sim = float(np.mean(v_sim[sub_mask_sim]))
    else:
        vrest_sim = float(np.mean(v_sim[:max(10, n // 10)]))
    vrest_mismatch = vrest_sim - vrest_target

    return {
        "final_loss": best_loss,
        "n_sim_spikes": n_sim_spikes,
        "n_target_spikes": n_target_spikes,
        "pearson_r": float(corr),
        "model_spikes": model_spikes,
        "no_spikes": no_spikes,
        "wrong_firing_rate": wrong_fr,
        "broad_spikes": broad_spikes,
        "excessive_sag": False,
        "parameters_at_bounds": at_bounds,
        "subthreshold_r2": float(sub_r2),
        "vrest_sim_mV": float(vrest_sim),
        "vrest_target_mV": float(vrest_target),
        "vrest_mismatch_mV": float(vrest_mismatch),
    }


# ===========================================================================
# Stimulus Windowing and Baseline
# ===========================================================================

def window_to_main_stimulus(stimulus, target_v, dt, max_duration_ms=1200.0,
                            pre_stim_ms=50.0):
    """Window stimulus and target to the main stimulus epoch.

    Includes pre_stim_ms of baseline BEFORE stimulus onset so the optimizer
    can see (and penalize) the model's resting potential mismatch. Without
    this, the trace starts at stimulus onset and the loss function never
    sees Vrest, causing eLeak to drift to whatever helps spike count."""
    stim_np = np.array(stimulus)
    stim_threshold = np.max(np.abs(stim_np)) * 0.1
    active = np.where(np.abs(stim_np) > stim_threshold)[0]

    if len(active) == 0:
        return stimulus, target_v, len(stimulus) * dt

    onset = active[0]
    offset = active[-1] + 1
    max_steps = int(max_duration_ms / dt)
    pre_steps = min(int(pre_stim_ms / dt), onset)  # don't go before trace start
    start = onset - pre_steps
    end = min(offset + int(50.0 / dt), start + max_steps, len(stim_np))

    stimulus_win = stimulus[start:end]
    target_win = target_v[start:end]
    t_max = len(stimulus_win) * dt

    return stimulus_win, target_win, t_max


def extract_baseline(stimulus_full, target_v_full, dt):
    """Extract pre-stimulus baseline voltage."""
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
# Main Entry Point: fit_proposal (with multi-start probe)
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
    n_starts: int = 5,
    warm_start_params: dict = None,
    n_sweeps: int = 1,
) -> DiagnosticReport:
    """
    Fit an agent-proposed model to a neuron's recordings.

    Uses multi-start probing + phased loss training:
        Probe: K diverse inits × n_probe_epochs (Phase 1 only) → pick best
        Phase 1 (remaining to phase_switch): spike_count + stats + MSE
        Phase 2 (phase_switch to end): adds spike_timing

    The multi-start probe eliminates initialization sensitivity by sampling
    across the spiking/non-spiking bifurcation boundary and selecting the
    initialization most likely to be in the spiking basin.

    Args:
        warm_start_params: dict of {param_name: float} from a previous best
            proposal's fitted_params. If provided, used as Start 0 in the
            multi-start probe (replacing the LLM's init as the baseline).
            This guarantees at least one probe starts from a known-spiking
            configuration when revising a working proposal.
    """
    data_dir = Path(data_dir)
    logger.info(f"  fit_proposal: channels={proposal.channels}, "
                f"specimen={specimen_id}, epochs={epochs}, n_starts={n_starts}")

    # Step 1: Load training data
    try:
        ctc = CellTypesCache(manifest_file=str(data_dir / "manifest.json"))
        with open(data_dir / "sweep_index.json") as f:
            sweep_index = json.load(f)
 
        if n_sweeps > 1:
            # Multi-sweep mode
            from multi_sweep_fitting import load_and_prepare_sweeps
            sweep_data_list = load_and_prepare_sweeps(
                ctc, specimen_id, sweep_index, dt=dt,
                max_duration_ms=max_duration_ms, n_sweeps=n_sweeps)
            logger.info(f"  Loaded {len(sweep_data_list)} sweeps for multi-sweep fitting")
 
            # Use the sweep with the MOST spikes for baseline/features/safety
            # This is the most informative sweep for gradient safety limits
            # and trace feature extraction (firing rate, half-width, etc.)
            primary_sweep = max(
                sweep_data_list,
                key=lambda sd: sd["shared"].get("raw_target_spike_count", 0)
            )
 
            # For cell construction: use primary sweep's features
            baseline_v = primary_sweep["baseline_v"]
            stimulus = primary_sweep["stimulus"]
            target_v_jnp = primary_sweep["target_v"]
            t_max = primary_sweep["t_max"]
            logger.info(f"  Primary sweep for features: #{primary_sweep['sweep_number']} "
                        f"({primary_sweep['stimulus_amplitude']:.0f} pA, "
                        f"{primary_sweep['shared'].get('raw_target_spike_count', 0)} spikes)")
        else:
            # Single-sweep mode (existing behavior, unchanged)
            sweep = load_training_sweep(ctc, specimen_id, sweep_index)
            stimulus, t_max = prepare_stimulus(sweep, dt)
            target_v = prepare_target(sweep, dt)
            target_v_jnp = jnp.array(target_v)
            sweep_data_list = None  # signals single-sweep mode
 
    except Exception as e:
        logger.error(f"  Data loading failed: {e}")
        return DiagnosticReport(proposal=proposal, specimen_id=specimen_id,
                                final_loss=float("inf"), no_spikes=True)
 
    # Step 2: Extract baseline, then window (single-sweep only; multi already windowed)
    if sweep_data_list is None:
        baseline_v = extract_baseline(stimulus, target_v, dt)
        logger.info(f"  Baseline: {len(baseline_v)} samples, Vrest={np.mean(baseline_v):.1f} mV")
 
        stimulus, target_v_jnp, t_max = window_to_main_stimulus(
            stimulus, target_v_jnp, dt, max_duration_ms)
        logger.info(f"  Stimulus window: {len(stimulus)} steps, {t_max:.0f} ms, "
                    f"stim range [{np.min(stimulus):.3f}, {np.max(stimulus):.3f}] nA")
    else:
        logger.info(f"  Baseline: {len(baseline_v)} samples, Vrest={np.mean(baseline_v):.1f} mV")

    # Step 3: Compute gradient safety limits
    target_np = np.array(target_v_jnp)
    try:
        features = extract_trace_features(target_np, dt, baseline_v=baseline_v)
        adaptive_limits = get_adaptive_hard_limits(features)
    except Exception as e:
        logger.warning(f"  Feature extraction failed ({e}), using static safety limits")
        adaptive_limits = None

    # Step 4: Build cell
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
        if sweep_data_list is not None and len(sweep_data_list) > 1:
            # Multi-sweep: register a ZERO stimulus to set the simulation
            # duration and recording. The actual per-sweep stimuli are injected
            # via data_stimulate() in the loss function.
            #
            # CRITICAL: We must NOT use the real stimulus here because
            # data_stimulate() ADDS to (not replaces) the static externals
            # registered by cell.stimulate(). Using a real stimulus would
            # double-stimulate the cell, producing NaN gradients.
            zero_stim = jnp.zeros_like(primary_sweep["stimulus"])
            cell = setup_simulation(cell, zero_stim, dt,
                                    primary_sweep["t_max"])
        else:
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
    phase_switch = int(epochs * phase_fraction)
    pre_data_stimuli = None  # Only set in multi-sweep mode; used by probe scoring
 
    if sweep_data_list is not None and len(sweep_data_list) > 1:
        # Multi-sweep loss
        from multi_sweep_fitting import (
            _build_multisweep_phase1_loss_fn,
            _build_multisweep_phase2_loss_fn,
            prebuild_data_stimuli,
        )

        # CRITICAL: Pre-build data_stimuli in eager mode (outside JIT).
        # cell.data_stimulate() has side effects on cell internals that
        # corrupt the computation graph when called inside JIT-traced code,
        # producing NaN gradients on the backward pass. Pre-building here
        # gives us pure JAX data structures safe for closure capture.
        pre_data_stimuli = prebuild_data_stimuli(cell, sweep_data_list)

        # Log per-sweep target info
        for sd in sweep_data_list:
            tc = int(sd["shared"].get("raw_target_spike_count", 0))
            logger.info(f"    Sweep #{sd['sweep_number']}: "
                        f"target_spikes={tc}, "
                        f"{sd['stimulus_amplitude']:.0f} pA")

        # Use primary sweep (most spikes) for probe scoring
        primary_idx = max(range(len(sweep_data_list)),
                        key=lambda i: sweep_data_list[i]["shared"].get("raw_target_spike_count", 0))
        shared = sweep_data_list[primary_idx]["shared"]
        logger.info(f"  Loss (multi-sweep, {len(sweep_data_list)} sweeps): "
                    f"spike_count=300, stats=5, MSE=0.1, timing=100 (Phase 2)")

        loss_fn_p1 = _build_multisweep_phase1_loss_fn(
            cell, sweep_data_list, dt, transform, param_names,
            pre_data_stimuli=pre_data_stimuli)
        loss_fn_p2 = _build_multisweep_phase2_loss_fn(
            cell, sweep_data_list, dt, transform, param_names,
            spike_timing_weight=100.0,
            pre_data_stimuli=pre_data_stimuli)

        # Diagnostic: run one forward pass to verify loss is computable
        try:
            diag_loss = loss_fn_p1(opt_params)
            diag_loss_f = float(diag_loss)
            logger.info(f"  Multi-sweep diagnostic forward pass: loss={diag_loss_f:.4f}, "
                        f"finite={np.isfinite(diag_loss_f)}")
            if not np.isfinite(diag_loss_f):
                # Run per-sweep sims to identify which sweep fails
                real_p = transform.forward(opt_params)
                ps = None
                for i, name in enumerate(param_names):
                    ps = cell.data_set(name, real_p[i][name], ps)
                for si, sd in enumerate(sweep_data_list):
                    try:
                        # Use pre-built data_stimuli — do NOT call
                        # cell.data_stimulate() here, which would add
                        # more external registrations to the cell.
                        v_dbg = jx.integrate(cell, param_state=ps,
                                             data_stimuli=pre_data_stimuli[si],
                                             delta_t=dt)
                        v_np = np.array(v_dbg[0])
                        logger.info(f"    Sweep #{sd['sweep_number']}: "
                                    f"v=[{np.min(v_np):.1f}, {np.max(v_np):.1f}], "
                                    f"nan={np.sum(np.isnan(v_np))}, "
                                    f"inf={np.sum(np.isinf(v_np))}")
                    except Exception as e2:
                        logger.error(f"    Sweep #{sd['sweep_number']}: sim failed: {e2}")
        except Exception as e:
            logger.warning(f"  Multi-sweep diagnostic forward pass failed: {e}")
    else:
        # Single-sweep loss (existing behavior)
        shared = _build_shared_loss_components(target_v_jnp, dt)
        logger.info(f"  Loss: target_spikes={int(shared['raw_target_spike_count'])}, "
                    f"spike_count=300, stats=5, MSE=0.1, timing=100 (Phase 2)")
 
        loss_fn_p1 = _build_phase1_loss_fn(cell, target_v_jnp, dt, transform,
                                            param_names, shared)
        loss_fn_p2 = _build_phase2_loss_fn(cell, target_v_jnp, dt, transform,
                                            param_names, shared,
                                            spike_timing_weight=100.0)

    # Step 7: Set up optimizer
    n_extra_channels = max(0, len(proposal.channels) - 3)
    stiffness_factor = 1.0 / (1.0 + 0.5 * n_extra_channels)
    lr_effective = lr * stiffness_factor

    clip_norm = 5.0 * np.sqrt(8.0 / max(len(param_names), 1))
    clip_norm = max(clip_norm, 1.0)

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
    optimizer = optax.chain(optax.clip_by_global_norm(clip_norm),
                            optax.adam(learning_rate=schedule))
    opt_state = optimizer.init(opt_params)

    logger.info(f"  Optimizer: lr={lr_effective:.4f}, clip={clip_norm:.2f}, "
                f"warmup={warmup_epochs}")

    # Step 8: JIT-compiled training steps
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

    # Phase switch: P1 (spike count only) → P2 (spike count + timing)
    # No warmup — if Phase 1's spike loss from the cleanup produces smoother
    # gradients, the timing loss can be added at full strength directly.

    # ===================================================================
    # Step 8b: MULTI-START PROBE (NEW)
    # ===================================================================
    n_probe_epochs = min(40, phase_switch // 3)  # Don't exceed 1/3 of Phase 1
    n_probe_epochs = max(20, n_probe_epochs)     # At least 20 epochs

    if n_starts > 1:
        starts = _generate_diverse_inits(proposal, trainable, n_starts=n_starts,
                                         warm_start_params=warm_start_params)
        logger.info(f"  Multi-start probe: {len(starts)} starts × {n_probe_epochs} epochs"
                     f"{' (with warm-start from previous best)' if warm_start_params else ''}")

        # Build a separate optimizer for probes (fresh LR schedule each time)
        probe_schedule = optax.cosine_decay_schedule(
            init_value=lr_effective, decay_steps=n_probe_epochs, alpha=0.1)
        probe_optimizer = optax.chain(
            optax.clip_by_global_norm(clip_norm),
            optax.adam(learning_rate=probe_schedule))

        # JIT a probe step with the probe optimizer
        @jit
        def probe_step(params, opt_state):
            loss_val, grads = value_and_grad(loss_fn_p1)(params)
            grad_finite = jax.tree.reduce(
                lambda a, b: a & b,
                jax.tree.map(lambda g: jnp.all(jnp.isfinite(g)), grads),
            )
            safe_grads = jax.tree.map(
                lambda g: jnp.where(grad_finite, g, jnp.zeros_like(g)), grads)
            updates, new_opt_state = probe_optimizer.update(safe_grads, opt_state, params)
            new_params = optax.apply_updates(params, updates)
            return new_params, new_opt_state, loss_val, grad_finite

        probe_results = []
        for start_idx, (label, overrides) in enumerate(starts):
            if overrides:
                try:
                    probe_params = _apply_init_overrides(
                        opt_params, transform, trainable, overrides)
                except Exception as e:
                    logger.warning(f"    Start {start_idx} ({label}): "
                                   f"override failed ({e}), skipping")
                    continue
            else:
                probe_params = jax.tree.map(lambda x: x.copy(), opt_params)

            probe_loss, probe_best, n_finite = _run_probe(
                probe_params, probe_optimizer, probe_step, n_probe_epochs)

            probe_results.append({
                "idx": start_idx, "label": label,
                "loss": probe_loss, "best_params": probe_best,
                "n_finite": n_finite,
            })

            logger.info(f"    Start {start_idx} ({label:>10s}): "
                         f"loss={probe_loss:.4f}, "
                         f"finite_grads={n_finite}/{n_probe_epochs}")

        # Select winner by lowest loss
        viable = [r for r in probe_results if r["best_params"] is not None]

        if not viable:
            logger.error("  All multi-start probes failed")
            return DiagnosticReport(proposal=proposal, specimen_id=specimen_id,
                                    final_loss=float("inf"), no_spikes=True)

        winner = min(viable, key=lambda r: r["loss"])

        logger.info(f"  Probe winner: Start {winner['idx']} ({winner['label']}) "
                     f"loss={winner['loss']:.4f}")

        # Continue from winner
        opt_params = jax.tree.map(lambda x: x.copy(), winner["best_params"])
        opt_state = optimizer.init(opt_params)
        best_loss = winner["loss"]
        best_params = jax.tree.map(lambda x: x.copy(), winner["best_params"])

        # Adjust phase boundaries (probe consumed some Phase 1 budget)
        effective_start_epoch = n_probe_epochs
    else:
        # Single start (n_starts=1): no probing, original behavior
        best_loss, best_params = float("inf"), None
        effective_start_epoch = 0

    # ===================================================================
    # Step 9: Training loop (continues from probe winner or original)
    # ===================================================================
    losses = []
    nan_count, max_nan, nan_grad_count = 0, 15, 0
    patience, epochs_since_best = 50, 0
    divergence_threshold = 3.0
    jitter_scale = 0.01
    rng = np.random.RandomState(42)
    current_phase = 1

    logger.info(f"  Training: epochs [{effective_start_epoch}, {epochs}), "
                f"phase_switch at {phase_switch}, lr={lr_effective:.4f}")

    for epoch in range(effective_start_epoch, epochs):
        # Phase switching: P1 → P2
        if epoch >= phase_switch and current_phase == 1:
            current_phase = 2
            logger.info(f"  === PHASE 2 at epoch {epoch}: "
                        f"adding spike_timing_weight=100.0 ===")
            best_loss = float("inf")
            epochs_since_best = 0

        step_fn = step_phase1 if current_phase == 1 else step_phase2

        try:
            opt_params, opt_state, loss_val, grad_ok = step_fn(opt_params, opt_state)
            loss_float = float(loss_val)
            grad_was_finite = bool(grad_ok)
        except Exception:
            nan_count += 1
            if nan_count >= max_nan:
                break
            if best_params is not None:
                opt_params = jax.tree.map(
                    lambda x: x + jitter_scale * jax.numpy.array(
                        np.array(rng.randn(*x.shape), dtype=x.dtype)),
                    best_params)
                opt_state = optimizer.init(opt_params)
            continue

        if np.isnan(loss_float) or np.isinf(loss_float):
            nan_count += 1
            if nan_count >= max_nan:
                break
            if best_params is not None:
                opt_params = jax.tree.map(
                    lambda x: x + jitter_scale * jax.numpy.array(
                        np.array(rng.randn(*x.shape), dtype=x.dtype)),
                    best_params)
                opt_state = optimizer.init(opt_params)
            if nan_count > 3:
                jitter_scale = min(jitter_scale * 1.5, 0.1)
            continue
        else:
            nan_count = 0
            jitter_scale = 0.01

        if not grad_was_finite:
            nan_grad_count += 1
            if nan_grad_count % 10 == 0:
                logger.info(f"    Epoch {epoch}: NaN gradient (zeroed), "
                            f"total={nan_grad_count}")
            continue

        losses.append(loss_float)
        if loss_float < best_loss:
            best_loss = loss_float
            best_params = jax.tree.map(lambda x: x.copy(), opt_params)
            epochs_since_best = 0
        else:
            epochs_since_best += 1

        if (best_params is not None and best_loss > 0
                and loss_float > best_loss * divergence_threshold):
            opt_params = jax.tree.map(lambda x: x.copy(), best_params)
            opt_state = optimizer.init(opt_params)
            epochs_since_best = 0

        if epochs_since_best >= patience and epoch > warmup_epochs + patience:
            logger.info(f"    Early stopping: no improvement for {patience} "
                        f"epochs (best={best_loss:.4f})")
            break

        if epoch % 20 == 0 or epoch == epochs - 1:
            phase_tag = f"[P{current_phase}]"
            logger.info(f"    Epoch {epoch:4d}  loss={loss_float:.4f}  "
                        f"best={best_loss:.4f}  {phase_tag}")

    # Step 10: Extract fitted parameters
    if best_params is None:
        logger.error("  No valid parameters found during optimisation")
        return DiagnosticReport(proposal=proposal, specimen_id=specimen_id,
                                final_loss=float("inf"), no_spikes=True)

    fitted = transform.forward(best_params)
    fitted_dict = {}
    for i, name in enumerate(param_names):
        val = fitted[i][name]
        fitted_dict[name] = (float(np.asarray(val).flatten()[0])
                             if isinstance(val, (list, np.ndarray, jnp.ndarray))
                             else float(val))

    logger.info(f"  Fitted parameters: {fitted_dict}")
    logger.info(f"  Best loss: {best_loss:.4f}")

    # Step 11: Final simulation (single-sweep only; multi-sweep does this in diagnostics)
    if sweep_data_list is not None and len(sweep_data_list) > 1:
        v_sim = None  # diagnostics will handle this
    else:
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

    # Step 12: Diagnostics
    if sweep_data_list is not None and len(sweep_data_list) > 1:
        # Multi-sweep diagnostics
        from multi_sweep_fitting import compute_multisweep_diagnostics
        diag = compute_multisweep_diagnostics(
            cell, sweep_data_list, dt, transform, param_names,
            best_params, proposal, trainable, best_loss)
 
        # For multi-sweep, also compute parameters_at_bounds
        at_bounds = []
        for t in trainable:
            name = t["name"]
            if name in fitted_dict:
                val = fitted_dict[name]
                rng = t["upper"] - t["lower"]
                if rng > 0:
                    if (val - t["lower"]) / rng < 0.02:
                        at_bounds.append(f"{name}=lower")
                    elif (t["upper"] - val) / rng < 0.02:
                        at_bounds.append(f"{name}=upper")
        diag["parameters_at_bounds"] = at_bounds
    else:
        # Single-sweep diagnostics (existing)
        target_np = np.array(target_v_jnp)
        diag = compute_diagnostics(v_sim, target_np, dt, proposal, fitted_dict,
                                   trainable, best_loss)
    
    proposal.fitted_params = fitted_dict
    proposal.loss = best_loss
    logger.info(f"  diag type: {type(diag)}, keys: {diag.keys() if isinstance(diag, dict) else 'NOT A DICT'}")
    report = DiagnosticReport(
        proposal=proposal, specimen_id=specimen_id,
        final_loss=best_loss,
        n_sim_spikes=diag.get("n_sim_spikes", 0),
        n_target_spikes=diag.get("n_target_spikes", 0),
        pearson_r=diag.get("pearson_r", 0.0),
        model_spikes=diag.get("model_spikes", False),
        no_spikes=diag.get("no_spikes", True),
        wrong_firing_rate=diag.get("wrong_firing_rate", False),
        broad_spikes=diag.get("broad_spikes", False),
        excessive_sag=diag.get("excessive_sag", False),
        parameters_at_bounds=diag.get("parameters_at_bounds", []),
    )

    logger.info(f"  Result: loss={best_loss:.4f}, "
                f"spikes={diag.get('n_sim_spikes', 0)}/{diag.get('n_target_spikes', 0)}, "
                f"r={diag.get('pearson_r', 0):.3f}")

    return report

# ===========================================================================
# CLI
# ===========================================================================

if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="cell_types_data")
    parser.add_argument("--specimen-id", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--channels", nargs="+",
                        default=["Na", "K", "Leak", "Kv3"])
    parser.add_argument("--phase-fraction", type=float, default=0.5)
    parser.add_argument("--n-starts", type=int, default=5,
                        help="Number of multi-start probes (1=disable)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if args.specimen_id is None:
        with open(data_dir / "sweep_index.json") as f:
            sweep_index = json.load(f)
        valid = [int(sid) for sid, e in sweep_index.items()
                 if e.get("valid")]
        specimen_id = valid[0] if valid else None
    else:
        specimen_id = args.specimen_id

    test_proposal = ModelProposal(
        proposal_id=0, iteration=0, channels=args.channels,
        param_config={}, rationale=f"CLI test: {args.channels}")
    report = fit_proposal(
        test_proposal, specimen_id=specimen_id,
        data_dir=str(data_dir), epochs=args.epochs,
        phase_fraction=args.phase_fraction,
        n_starts=args.n_starts)
    print(report.generate_feedback())