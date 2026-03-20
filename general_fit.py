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
#
# UNITS: S/cm² for conductances (Jaxley/NEURON convention).
# Reference values from Pospischil et al. (2008) Table 1:
#   FS cell:  gNa=0.058, gKd=0.0039, gLeak=3.8e-5, gM=0 S/cm²
#   RS cell:  gNa=0.050, gKd=0.005,  gLeak=1.0e-5, gM=7e-5 S/cm²
#
# Init values are centered near Pospischil FS parameters.
# Bounds span roughly 0.1x–10x the typical value to give the optimizer
# room without leaving the biophysically plausible regime.
# Previous bounds had inits 100–2000× too large (e.g. Na_gNa init=0.5 vs
# typical 0.05), causing massive currents, depolarization block, and NaN.
DEFAULT_PARAM_BOUNDS = {
    # --- Built-in channel conductances ---
    "Na_gNa": {
        "init": 0.10,           # Raised from 0.05: with Kv3 present, 0.05 produces
                                # a non-spiking model with NaN gradients everywhere.
                                # 0.10 is mid-range and guarantees initial spiking.
        "lower": 0.01,
        "upper": 0.20,
    },
    "K_gK": {
        "init": 0.005,          # Jaxley default
        "lower": 0.001,
        "upper": 0.05,
    },
    "Leak_gLeak": {
        "init": 0.0001,         # Jaxley default
        "lower": 1e-5,
        "upper": 0.002,
    },
    "Leak_eLeak": {
        "init": -70.0,          # Jaxley default (mV)
        "lower": -85.0,         # Widened from -80: optimizer consistently pins here
        "upper": -50.0,
    },
 
    # --- Custom channel conductances (from channels.py) ---
    "Kv3_gKv3": {
        "init": 0.005,          # Raised from 0.003: closer to mid-range of bounds
        "lower": 5e-4,
        "upper": 0.03,
    },
    "IM_gM": {
        "init": 7e-5,           # Pospischil RS value (S/cm²)
        "lower": 1e-6,
        "upper": 0.005,
    },
    "IAHP_gAHP": {
        "init": 1e-4,
        "lower": 1e-6,
        "upper": 0.005,
    },
    "IT_gT": {
        "init": 1e-4,
        "lower": 1e-6,
        "upper": 0.005,
    },
    "ICaL_gCaL": {
        "init": 1e-4,
        "lower": 1e-6,
        "upper": 0.005,
    },
    "IH_gH": {
        "init": 2e-5,
        "lower": 1e-6,
        "upper": 0.001,
    },
 
    # --- Global parameters ---
    "eNa": {
        "init": 55.0,           # Raised from 50: gives more Na driving force
        "lower": 40.0,
        "upper": 70.0,          # Widened from 65: optimizer pins at 65 consistently
    },
    "eK": {
        "init": -90.0,          # Jaxley default (mV)
        "lower": -110.0,        # Widened from -100: optimizer pins at -100 consistently
        "upper": -70.0,
    },
    "capacitance": {
        "init": 1.0,            # µF/cm²
        "lower": 0.5,
        "upper": 2.0,
    },
    "radius": {
        "init": 10.0,           # µm
        "lower": 3.0,
        "upper": 20.0,
    },
}

# Maximum factor by which LLM-proposed bounds can exceed defaults.
# 5× is generous enough for the optimizer but prevents NaN-causing extremes
# like Na_gNa=120 S/cm² (which should be ~0.05).
_LLM_BOUND_CLAMP_FACTOR = 5.0

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

# Maximum allowed upper bounds per parameter (gradient-safe ceilings).
# These were determined empirically: values above these can cause NaN
# in backprop-through-time on 46,000-step traces with float32.
_HARD_UPPER_CEILINGS = {
    "Na_gNa":     0.25,     # tested: 0.2 OK, 0.5 NaN
    "K_gK":       0.08,     # tested: 0.05 OK, 0.2 NaN
    "Leak_gLeak": 0.005,    # too much leak kills spiking
    "Kv3_gKv3":   0.08,     # same order as K
    "IM_gM":      0.01,
    "IAHP_gAHP":  0.01,
    "IT_gT":      0.01,
    "ICaL_gCaL":  0.01,
    "IH_gH":      0.005,
}
 
_HARD_LOWER_FLOORS = {
    "eNa": 35.0,
    "eK": -110.0,
    "capacitance": 0.3,
    "radius": 2.0,
}
 
_HARD_UPPER_CEILINGS_GLOBAL = {
    "eNa": 80.0,            # Widened from 70: iter 2 pinned at 65, iter 3 at 70
    "eK": -65.0,
    "capacitance": 3.0,
    "radius": 25.0,
}
 
 
def _clamp_param_bounds(name: str, cfg: dict) -> dict:
    """
    Clamp LLM-proposed parameter bounds to the gradient-safe regime.
 
    The LLM may propose arbitrarily wide bounds (e.g., gNa upper = 120).
    This function enforces hard ceilings determined by empirical gradient
    stability testing on 46,000-step Jaxley simulations with float32.
 
    Strategy:
      - Conductances: enforce absolute ceiling from _HARD_UPPER_CEILINGS
      - Reversal potentials: enforce absolute floor/ceiling
      - Geometry: enforce absolute floor/ceiling
      - Init values: clamp to [lower, upper] after bound clamping
 
    Returns a new dict (does not mutate input).
    """
    cfg = dict(cfg)  # don't mutate caller's dict
 
    # --- Conductance parameters: hard upper ceiling ---
    if name in _HARD_UPPER_CEILINGS:
        ceiling = _HARD_UPPER_CEILINGS[name]
        if cfg.get("upper", ceiling) > ceiling:
            logger.warning(f"  Clamping {name} upper: {cfg['upper']} -> {ceiling}")
            cfg["upper"] = ceiling
        # Also enforce that lower >= some minimum
        default = DEFAULT_PARAM_BOUNDS.get(name, {})
        min_lower = default.get("lower", cfg.get("lower", 0))
        if cfg.get("lower", 0) < min_lower:
            cfg["lower"] = min_lower
 
    # --- Global parameters: hard floor/ceiling ---
    if name in _HARD_LOWER_FLOORS:
        floor = _HARD_LOWER_FLOORS[name]
        if cfg.get("lower", floor) < floor:
            logger.warning(f"  Clamping {name} lower: {cfg['lower']} -> {floor}")
            cfg["lower"] = floor
 
    if name in _HARD_UPPER_CEILINGS_GLOBAL:
        ceiling = _HARD_UPPER_CEILINGS_GLOBAL[name]
        if cfg.get("upper", ceiling) > ceiling:
            logger.warning(f"  Clamping {name} upper: {cfg['upper']} -> {ceiling}")
            cfg["upper"] = ceiling
 
    # --- Ensure init is within [lower, upper] ---
    if "init" in cfg:
        cfg["init"] = max(cfg["init"], cfg.get("lower", cfg["init"]))
        cfg["init"] = min(cfg["init"], cfg.get("upper", cfg["init"]))
 
    # --- Sanity: lower < upper ---
    if cfg.get("lower", 0) >= cfg.get("upper", float("inf")):
        logger.warning(f"  {name}: lower >= upper ({cfg['lower']} >= {cfg['upper']}), "
                       f"resetting to defaults")
        default = DEFAULT_PARAM_BOUNDS.get(name, cfg)
        cfg["lower"] = default.get("lower", cfg["lower"])
        cfg["upper"] = default.get("upper", cfg["upper"])
 
    return cfg

def _ensure_init_margin(init_val, lower, upper, margin_frac=0.05):
    """
    Ensure init value is at least margin_frac away from sigmoid bounds.
 
    When init == upper or init == lower, the SigmoidTransform's logit
    maps to ±infinity, causing immediate NaN in the gradient.
 
    Args:
        init_val: proposed initial value
        lower, upper: sigmoid bounds
        margin_frac: fraction of range to keep as margin (default 5%)
 
    Returns:
        Clamped init value guaranteed to be inside [lower + margin, upper - margin]
    """
    margin = (upper - lower) * margin_frac
    return float(max(lower + margin, min(upper - margin, init_val)))

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
        comp.set("eK", -90.0)

        # Set initial conductance values from proposal or defaults
        for ch_name in channels:
            for param_name in CHANNEL_CONDUCTANCE_PARAMS.get(ch_name, []):
                if param_name in proposal.param_config:
                    init_val = _clamp_param_bounds(
                        param_name, proposal.param_config[param_name]
                    ).get("init")
                elif param_name in DEFAULT_PARAM_BOUNDS:
                    cfg = DEFAULT_PARAM_BOUNDS[param_name]
                    init_val = _ensure_init_margin(cfg["init"], cfg["lower"], cfg["upper"])
                else:
                    continue
                try:
                    comp.set(param_name, init_val)
                except Exception as e:
                    logger.warning(f"  Could not set {param_name}={init_val}: {e}")

    except Exception as e:
        return None, None, f"Cell construction failed: {e}\n{traceback.format_exc()}"

    # ------------------------------------------------------------------
    # Collect trainable parameters with CLAMPED bounds
    # ------------------------------------------------------------------
    trainable = []

    # Channel-specific conductance params
    for ch_name in channels:
        for param_name in CHANNEL_CONDUCTANCE_PARAMS.get(ch_name, []):
            if param_name in proposal.param_config:
                cfg = _clamp_param_bounds(param_name, proposal.param_config[param_name])
            elif param_name in DEFAULT_PARAM_BOUNDS:
                cfg = DEFAULT_PARAM_BOUNDS[param_name]
            else:
                continue
            trainable.append({
                "name": param_name,
                "lower": cfg["lower"],
                "upper": cfg["upper"],
            })

    # Global params (eNa, eK, capacitance, radius) — ALSO clamped
    for param_name in GLOBAL_TRAINABLE:
        if param_name not in DEFAULT_PARAM_BOUNDS:
            continue
        if param_name in proposal.param_config:
            cfg = _clamp_param_bounds(param_name, proposal.param_config[param_name])
        else:
            cfg = DEFAULT_PARAM_BOUNDS[param_name]
        trainable.append({
            "name": param_name,
            "lower": cfg["lower"],
            "upper": cfg["upper"],
        })

    return comp, trainable, None


# ===========================================================================
# Build Loss Function (generalized)
# ===========================================================================

def build_generalized_loss_fn(cell, target_v, dt, transform, param_names,
                              n_windows=10, mse_weight=0.1,
                              stats_weight=5.0,
                              spike_count_weight=300.0,
                              spike_timing_weight=100.0,
                              spike_threshold=-20.0):
    """
    Build a differentiable loss function combining:
      1. Waveform MSE (downweighted — refinement term)
      2. Windowed summary statistics (Deistler et al. 2025)
      3. Differentiable spike count penalty
      4. Smoothed spike train distance (spike timing)

    LOSS HIERARCHY (by design):
      spike_count >> spike_timing ≈ windowed_stats >> MSE

    SPIKE TIMING (Component 4):
      A differentiable van Rossum-like distance. Both sim and target
      voltage traces are converted to soft spike event trains via
      sigmoid threshold crossings, then smoothed with a Gaussian kernel
      (σ = 2 ms). MSE between the smoothed trains provides direct
      gradient toward aligning individual spike times:
        - If a sim spike is 3ms late, the Gaussians overlap partially,
          and the gradient pushes parameters to shift the spike earlier.
        - If a spike is missing, the unmatched target Gaussian creates
          gradient toward producing a spike at that time.
        - σ = 2 ms sets the timing tolerance: spikes within ~4ms are
          considered well-aligned; beyond ~6ms they contribute fully.

      This complements the spike count term (which only cares about
      total count, not where spikes occur) and the windowed stats
      (which are too coarse at 115ms windows to resolve individual
      spike positions).

    Args:
        n_windows:           Number of temporal windows for summary stats.
        mse_weight:          Weight of raw MSE (refinement signal).
        stats_weight:        Weight of windowed stats (mean + std per window).
        spike_count_weight:  Weight of spike count loss.
        spike_timing_weight: Weight of smoothed spike train distance.
        spike_threshold:     Voltage threshold for spike detection (mV).
    """
    # Pre-compute target summary statistics (non-differentiable, constants)
    target_np = np.array(target_v)
    n_total = len(target_np)
    win_size = n_total // n_windows

    # Pre-compute target spike count
    tgt_crossings = np.diff((target_np > spike_threshold).astype(int)) > 0
    target_spike_count = max(float(np.sum(tgt_crossings)), 1.0)

    logger.info(f"  Loss function: target_spikes={int(target_spike_count)}, "
                f"mse_weight={mse_weight}, stats_weight={stats_weight}, "
                f"spike_count_weight={spike_count_weight}, "
                f"spike_timing_weight={spike_timing_weight}")

    target_means = []
    target_stds = []
    for w in range(n_windows):
        start = w * win_size
        end = start + win_size if w < n_windows - 1 else n_total
        window = target_np[start:end]
        target_means.append(float(np.mean(window)))
        target_stds.append(float(np.std(window)))

    tgt_means = jnp.array(target_means)
    tgt_stds = jnp.array(target_stds)

    mean_scale = 8.0
    std_scale = 4.0

    # --- Spike timing kernel setup ---
    # Gaussian kernel: σ = 2 ms, truncated at ±3σ
    spike_k = 0.5  # sigmoid sharpness for soft spike detection
    sigma_ms = 2.0
    sigma_steps = sigma_ms / dt
    half_kernel = int(3.0 * sigma_steps)
    kernel_x = jnp.arange(-half_kernel, half_kernel + 1, dtype=jnp.float32)
    timing_kernel = jnp.exp(-kernel_x ** 2 / (2.0 * sigma_steps ** 2))
    timing_kernel = timing_kernel / jnp.sum(timing_kernel)  # normalise to sum=1

    logger.info(f"  Spike timing kernel: sigma={sigma_ms}ms ({sigma_steps:.0f} steps), "
                f"kernel length={len(timing_kernel)}")

    # Pre-compute smoothed target spike train (constant)
    # Use the same soft detection as sim for consistent scale
    tgt_p = 1.0 / (1.0 + np.exp(-spike_k * (target_np - spike_threshold)))
    tgt_dp = np.diff(tgt_p)
    tgt_events = np.maximum(tgt_dp, 0.0)  # upward crossings only
    tgt_smoothed = jnp.array(
        np.convolve(tgt_events, np.array(timing_kernel), mode='same'),
        dtype=jnp.float32,
    )
    # Normalise so peak ~ 1.0, making the MSE scale interpretable
    tgt_peak = float(np.max(tgt_smoothed)) if float(np.max(tgt_smoothed)) > 0 else 1.0
    tgt_smoothed = tgt_smoothed / tgt_peak

    def loss_fn(opt_params):
        params = transform.forward(opt_params)

        param_state = None
        for i, name in enumerate(param_names):
            param_state = cell.data_set(name, params[i][name], param_state)

        try:
            v = jx.integrate(cell, param_state=param_state, delta_t=dt)
            v_sim = v[0]

            n = min(len(v_sim), len(target_v))
            v_s = v_sim[:n]
            v_t = target_v[:n]

            # --- Component 1: Waveform MSE (refinement term) ---
            mse = jnp.mean((v_s - v_t) ** 2)

            # --- Component 2: Windowed summary statistics ---
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

            mean_loss = jnp.mean(jnp.abs(sim_means / mean_scale - tgt_means / mean_scale))
            std_loss = jnp.mean(jnp.abs(sim_stds / std_scale - tgt_stds / std_scale))

            stats_loss = mean_loss + std_loss

            # --- Component 3: Differentiable spike count loss ---
            p = jax.nn.sigmoid(spike_k * (v_s - spike_threshold))
            dp = jnp.diff(p)
            soft_events = jax.nn.relu(dp)
            soft_count = jnp.sum(soft_events)
            spike_loss = (soft_count / target_spike_count - 1.0) ** 2

            # --- Component 4: Smoothed spike train distance (timing) ---
            sim_smoothed = jnp.convolve(soft_events, timing_kernel, mode='same')
            sim_smoothed = sim_smoothed / tgt_peak  # same normalisation as target

            # MSE between smoothed spike trains
            n_timing = min(len(sim_smoothed), len(tgt_smoothed))
            timing_loss = jnp.mean(
                (sim_smoothed[:n_timing] - tgt_smoothed[:n_timing]) ** 2
            )

            total = (mse_weight * mse
                     + stats_weight * stats_loss
                     + spike_count_weight * spike_loss
                     + spike_timing_weight * timing_loss)

        except Exception:
            total = jnp.array(float("inf"))

        return total

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
            bound_range = t_info["upper"] - t_info["lower"]
            buffer = bound_range * 0.001  # 0.1% buffer prevents logit(0)/logit(1) → ±inf
            transforms.append({
                t_info["name"]: SigmoidTransform(
                    lower=t_info["lower"] - buffer,
                    upper=t_info["upper"] + buffer,
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

    # --- Stiffness-aware LR scaling ---
    # Extra channels beyond Na+K+Leak stiffen the ODE Jacobian.
    # Scale LR down to keep the first optimizer step safe.
    n_extra_channels = max(0, len(proposal.channels) - 3)  # beyond Na, K, Leak
    stiffness_factor = 1.0 / (1.0 + 0.5 * n_extra_channels)  # Kv3 → 0.67×
    lr_effective = lr * stiffness_factor
    logger.info(f"  LR scaling: base={lr}, stiffness_factor={stiffness_factor:.2f} "
                f"({n_extra_channels} extra channels), effective={lr_effective:.4f}")

    # --- Adaptive gradient clip norm ---
    # More parameters = gradient norm budget spread thinner per param.
    # Scale clip norm down with sqrt(n_params) relative to 8-param baseline.
    clip_norm = 5.0 * np.sqrt(8.0 / max(len(param_names), 1))
    clip_norm = max(clip_norm, 1.0)  # floor at 1.0
    logger.info(f"  Gradient clip norm: {clip_norm:.2f} (for {len(param_names)} params)")

    # --- Cosine LR schedule with linear warmup ---
    warmup_epochs = min(30, epochs // 5)  # longer warmup for stability
    schedule = optax.join_schedules(
        schedules=[
            optax.linear_schedule(init_value=lr_effective * 0.01,  # start at 1% (was 10%)
                                  end_value=lr_effective,
                                  transition_steps=warmup_epochs),
            optax.cosine_decay_schedule(init_value=lr_effective,
                                        decay_steps=epochs - warmup_epochs,
                                        alpha=0.01),
        ],
        boundaries=[warmup_epochs],
    )

    optimizer = optax.chain(
        optax.clip_by_global_norm(clip_norm),
        optax.adam(learning_rate=schedule),
    )
    opt_state = optimizer.init(opt_params)

    # JIT-compiled training step with NaN-safe gradient handling.
    # If any gradient component is NaN/inf, we replace the entire gradient
    # tree with zeros. This means "skip this step" rather than propagating
    # NaN through clip_by_global_norm (which would produce all-NaN updates
    # because norm(NaN) = NaN → scale = NaN).
    @jit
    def step(params, opt_state):
        loss_val, grads = value_and_grad(loss_fn)(params)

        # Check if any gradient leaf contains NaN/inf
        grad_finite = jax.tree.reduce(
            lambda a, b: a & b,
            jax.tree.map(lambda g: jnp.all(jnp.isfinite(g)), grads),
        )
        # If gradient has NaN, zero it out (no-op step, preserves opt_state)
        safe_grads = jax.tree.map(
            lambda g: jnp.where(grad_finite, g, jnp.zeros_like(g)),
            grads,
        )

        updates, new_opt_state = optimizer.update(safe_grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss_val, grad_finite
 
    losses = []
    best_loss = float("inf")
    best_params = None
    nan_count = 0
    max_nan = 15              # increased: rollback+jitter gives more recovery chances
    nan_grad_count = 0        # track NaN gradients separately
    patience = 50
    epochs_since_best = 0
    divergence_threshold = 3.0
    jitter_scale = 0.01       # scale of random perturbation on rollback
    rng = np.random.RandomState(42)

    logger.info(f"  Starting optimisation: {epochs} epochs, lr={lr_effective:.4f} "
                f"(cosine schedule, warmup={warmup_epochs})")

    for epoch in range(epochs):
        try:
            opt_params, opt_state, loss_val, grad_ok = step(opt_params, opt_state)
            loss_float = float(loss_val)
            grad_was_finite = bool(grad_ok)
        except Exception as e:
            logger.warning(f"  Epoch {epoch}: simulation error — {e}")
            nan_count += 1
            if nan_count >= max_nan:
                logger.error(f"  {max_nan} consecutive failures, stopping")
                break
            # Rollback with jitter to escape the unstable basin
            if best_params is not None:
                opt_params = jax.tree.map(
                    lambda x: x + jitter_scale * jax.numpy.array(
                        rng.randn(*x.shape).astype(np.float32)),
                    best_params,
                )
                opt_state = optimizer.init(opt_params)
                logger.info(f"    Epoch {epoch}: rollback + jitter (scale={jitter_scale:.3f})")
            continue

        # Handle NaN loss
        if np.isnan(loss_float) or np.isinf(loss_float):
            nan_count += 1
            if nan_count >= max_nan:
                logger.warning(f"  {max_nan} consecutive NaN/inf, stopping")
                break
            if best_params is not None:
                # Rollback with jitter to break the deterministic death spiral
                opt_params = jax.tree.map(
                    lambda x: x + jitter_scale * jax.numpy.array(
                        rng.randn(*x.shape).astype(np.float32)),
                    best_params,
                )
                opt_state = optimizer.init(opt_params)
                logger.info(f"    Epoch {epoch}: NaN loss, rollback + jitter")
            # Increase jitter progressively if NaN persists
            if nan_count > 3:
                jitter_scale = min(jitter_scale * 1.5, 0.1)
            continue
        else:
            nan_count = 0
            jitter_scale = 0.01  # reset jitter on successful step

        # Track NaN gradients (step was no-op due to zeroed grads)
        if not grad_was_finite:
            nan_grad_count += 1
            if nan_grad_count % 10 == 0:
                logger.info(f"    Epoch {epoch}: NaN gradient (zeroed), "
                            f"total={nan_grad_count}")
            # Loss is finite but gradient isn't — the step was a no-op,
            # so opt_state still advanced (Adam counters tick), which
            # naturally changes the next step. No rollback needed.
            continue

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
            epochs_since_best = 0

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