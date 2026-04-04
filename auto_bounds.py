"""
Trace Analysis & Gradient Safety for NASS
==========================================

Two responsibilities, cleanly separated from biophysical judgment:

  1. **Feature extraction** — measure objective properties of the target
     voltage trace (spike peaks, AHP depth, resting potential, firing rate,
     spike width). These features are passed to the LLM agent so it can
     make informed biophysical decisions about parameter bounds.

  2. **Gradient safety** — compute hard limits that prevent NaN in
     JAX backprop-through-time. These are numerical constraints about
     float32 stability, NOT biophysical judgment. They are the only
     programmatic override applied to the LLM's proposed bounds.

What this module does NOT do (the LLM agent's job):
  - Choose initial conductance values
  - Decide appropriate parameter ranges for a neuron type
  - Apply single-compartment corrections
  - Detect FS vs RS cells and adjust accordingly
  - Any other biophysical reasoning

Usage:
    from auto_bounds import extract_trace_features, get_adaptive_hard_limits

    # Extract features once, pass to LLM prompts
    features = extract_trace_features(windowed_trace, dt, baseline_v=pre_stim)

    # Compute gradient safety limits (applied after LLM proposes bounds)
    hard_limits = get_adaptive_hard_limits(features)
"""

import logging
import numpy as np
from typing import Optional

logger = logging.getLogger(__name__)


# ============================================================================
# PART 1: TRACE FEATURE EXTRACTION
# ============================================================================

def extract_trace_features(target_v: np.ndarray, dt: float,
                           threshold: float = -20.0,
                           min_isi_ms: float = 2.0,
                           baseline_v: np.ndarray = None) -> dict:
    """
    Extract electrophysiological features directly from the target voltage
    trace. These are objective measurements passed to the LLM agent.

    Args:
        target_v:   Target voltage trace (mV), windowed to stimulus period
        dt:         Timestep in ms
        threshold:  Spike detection threshold (mV)
        min_isi_ms: Minimum inter-spike interval for spike detection (ms)
        baseline_v: Pre-stimulus baseline voltage (mV). If provided, used
                    for true resting potential instead of first 10% of
                    target_v (which may be depolarized if windowed).

    Returns:
        dict of measured features
    """
    n = len(target_v)
    t_total_ms = n * dt

    # --- Resting potential ---
    if baseline_v is not None and len(baseline_v) > 10:
        resting_potential = float(np.mean(baseline_v))
    else:
        n_baseline = max(10, n // 10)
        resting_potential = float(np.mean(target_v[:n_baseline]))

    # --- Spike detection: upward threshold crossings ---
    above = target_v > threshold
    crossings = np.where(np.diff(above.astype(np.int32)) > 0)[0]

    if len(crossings) > 1:
        filtered = [crossings[0]]
        for idx in crossings[1:]:
            if (idx - filtered[-1]) * dt >= min_isi_ms:
                filtered.append(idx)
        crossings = np.array(filtered)

    n_spikes = len(crossings)
    firing_rate_hz = n_spikes / (t_total_ms / 1000.0) if t_total_ms > 0 else 0.0

    # --- Spike peak voltages ---
    spike_peaks = []
    ahp_troughs = []
    spike_halfwidths = []

    for i, onset_idx in enumerate(crossings):
        peak_window = int(5.0 / dt)
        end_peak = min(onset_idx + peak_window, n)
        peak_idx = onset_idx + np.argmax(target_v[onset_idx:end_peak])
        spike_peaks.append(float(target_v[peak_idx]))

        trough_start = peak_idx + int(2.0 / dt)
        trough_end = min(peak_idx + int(20.0 / dt), n)
        if trough_start < trough_end:
            trough_idx = trough_start + np.argmin(target_v[trough_start:trough_end])
            ahp_troughs.append(float(target_v[trough_idx]))

        if peak_idx < n:
            half_v = (target_v[peak_idx] + resting_potential) / 2.0
            hw_start = onset_idx
            for j in range(peak_idx, onset_idx, -1):
                if target_v[j] < half_v:
                    hw_start = j
                    break
            hw_end = min(peak_idx + int(5.0 / dt), n - 1)
            for j in range(peak_idx, hw_end):
                if target_v[j] < half_v:
                    hw_end = j
                    break
            hw_ms = (hw_end - hw_start) * dt
            if 0.1 < hw_ms < 10.0:
                spike_halfwidths.append(hw_ms)

    isis = []
    if len(crossings) > 1:
        isis = np.diff(crossings) * dt

    features = {
        "resting_potential_mV": round(resting_potential, 1),
        "spike_peak_mean_mV": round(float(np.mean(spike_peaks)), 1) if spike_peaks else round(resting_potential + 20, 1),
        "spike_peak_p90_mV": round(float(np.percentile(spike_peaks, 90)), 1) if spike_peaks else round(resting_potential + 30, 1),
        "ahp_trough_mean_mV": round(float(np.mean(ahp_troughs)), 1) if ahp_troughs else round(resting_potential - 10, 1),
        "ahp_trough_p10_mV": round(float(np.percentile(ahp_troughs, 10)), 1) if ahp_troughs else round(resting_potential - 15, 1),
        "n_spikes": n_spikes,
        "firing_rate_hz": round(firing_rate_hz, 1),
        "mean_isi_ms": round(float(np.mean(isis)), 2) if len(isis) > 0 else 0.0,
        "spike_halfwidth_ms": round(float(np.median(spike_halfwidths)), 3) if spike_halfwidths else 1.0,
        "voltage_min_mV": round(float(np.min(target_v)), 1),
        "voltage_max_mV": round(float(np.max(target_v)), 1),
        "cv_isi": round(float(np.std(isis) / np.mean(isis)), 3) if len(isis) > 1 else 0.0,
    }

    logger.info(f"  Trace features: {n_spikes} spikes, {features['firing_rate_hz']} Hz, "
                f"Vrest={resting_potential:.1f} mV, peak={features['spike_peak_mean_mV']:.1f} mV, "
                f"AHP={features['ahp_trough_mean_mV']:.1f} mV, hw={features['spike_halfwidth_ms']:.2f} ms")

    return features


def format_features_for_prompt(features: dict) -> str:
    """
    Format extracted trace features as a readable block for LLM prompts.
    """
    hw = features['spike_halfwidth_ms']
    rate = features['firing_rate_hz']
    is_fs = (hw < 0.8 and rate > 30) or (rate > 80)

    return f"""## Measured Trace Features (from target recording)
These are objective measurements from the target voltage trace.
Use them to set appropriate parameter bounds in your param_config.

  Resting potential:    {features['resting_potential_mV']} mV
  Spike peak (mean):    {features['spike_peak_mean_mV']} mV
  Spike peak (90th %):  {features['spike_peak_p90_mV']} mV
  AHP trough (mean):    {features['ahp_trough_mean_mV']} mV
  AHP trough (10th %):  {features['ahp_trough_p10_mV']} mV
  Number of spikes:     {features['n_spikes']}
  Firing rate:          {features['firing_rate_hz']} Hz
  Mean ISI:             {features['mean_isi_ms']} ms
  Spike half-width:     {features['spike_halfwidth_ms']} ms
  ISI CV:               {features['cv_isi']}
  Voltage range:        [{features['voltage_min_mV']}, {features['voltage_max_mV']}] mV

### Key constraints these features impose on your param_config:
- eNa MUST exceed spike peak ({features['spike_peak_p90_mV']} mV). For single-compartment
  models, eNa is typically 30-90 mV ABOVE the spike peak because Na channels
  inactivate before V reaches eNa. Typical range: 50-115 mV.
- eK MUST be below AHP trough ({features['ahp_trough_p10_mV']} mV). Typical range: -120 to -70 mV.
- **Leak_eLeak MUST be near resting potential ({features['resting_potential_mV']} mV)**.
  Set init={features['resting_potential_mV']}, bounds=[{features['resting_potential_mV']-10:.0f}, {features['resting_potential_mV']+10:.0f}].
  The optimizer penalizes pre-stimulus baseline mismatch heavily -- a 12 mV
  Vrest offset causes R2=0 on held-out sweeps. DO NOT use wide bounds like [-85, -50].
- Na_gNa init MUST be >= 0.10 S/cm² for single-compartment models to spike.
  Below this, Kv3 suppresses all spiking and gradients are NaN.
- {'This is a FAST-SPIKING cell (narrow spikes, high rate). Use Kv3, prefer small radius (5-12 µm), keep capacitance ≤ 2.0.' if is_fs else 'This appears to be a regular-spiking or adapting cell. Consider IM/IAHP for adaptation.'}"""


# ============================================================================
# PART 2: GRADIENT SAFETY LIMITS
# ============================================================================

def get_adaptive_hard_limits(features: dict) -> tuple:
    """
    Compute gradient-safety hard limits based on the cell's properties.

    These are NUMERICAL constraints about float32/BPTT stability, NOT
    biophysical judgment. They prevent the optimizer from entering
    parameter regimes that cause NaN in backprop.

    The NaN boundary depends on:
      - Number of spikes (more spikes → more gradient accumulation)
      - Simulation length (more steps → tighter limits)

    These limits are intentionally WIDE — they only catch truly extreme
    proposals. The LLM's biophysical judgment should keep parameters
    well within these limits in normal operation.

    Returns:
        (hard_ceilings, hard_floors, hard_ceiling_globals)
    """
    n_spikes = features.get("n_spikes", 50)

    # More spikes → more gradient accumulation → tighter conductance limits
    spike_factor = max(1.0, n_spikes / 50.0)
    scale = 1.0 / np.sqrt(spike_factor)

    # Conductance ceilings: generous but prevent NaN
    base_ceilings = {
        "Na_gNa":     0.50,
        "K_gK":       0.10,
        "Leak_gLeak": 0.005,
        "Kv3_gKv3":   0.10,
        "IM_gM":      0.01,
        "IAHP_gAHP":  0.01,
        "IT_gT":      0.01,
        "ICaL_gCaL":  0.01,
        "IH_gH":      0.005,
    }

    hard_ceilings = {k: max(v * scale, v * 0.3) for k, v in base_ceilings.items()}

    # Global parameter limits: wide numerical limits
    hard_floors = {
        "eNa": 30.0,
        "eK": -120.0,
        "capacitance": 0.2,
        "radius": 1.5,
    }

    hard_ceiling_globals = {
        "eNa": 115.0,
        "eK": -55.0,
        "capacitance": 3.0,
        "radius": 30.0,
    }

    return hard_ceilings, hard_floors, hard_ceiling_globals


def clamp_to_gradient_safety(name: str, cfg: dict,
                              hard_ceilings: dict,
                              hard_floors: dict,
                              hard_ceiling_globals: dict) -> dict:
    """
    Clamp LLM-proposed bounds to gradient-safe regime ONLY.

    No biophysical judgment — just NaN prevention.
    The LLM is trusted to make biophysically reasonable choices;
    this function only catches proposals that would crash the optimizer.
    """
    cfg = dict(cfg)

    if name in hard_ceilings:
        ceiling = hard_ceilings[name]
        if cfg.get("upper", ceiling) > ceiling:
            logger.warning(f"  Gradient safety: clamping {name} upper "
                           f"{cfg['upper']:.6f} -> {ceiling:.6f}")
            cfg["upper"] = ceiling

    if name in hard_floors:
        floor = hard_floors[name]
        if cfg.get("lower", floor) < floor:
            logger.warning(f"  Gradient safety: clamping {name} lower "
                           f"{cfg['lower']:.4f} -> {floor:.4f}")
            cfg["lower"] = floor

    if name in hard_ceiling_globals:
        ceiling = hard_ceiling_globals[name]
        if cfg.get("upper", ceiling) > ceiling:
            logger.warning(f"  Gradient safety: clamping {name} upper "
                           f"{cfg['upper']:.4f} -> {ceiling:.4f}")
            cfg["upper"] = ceiling

    if "init" in cfg:
        cfg["init"] = max(cfg["init"], cfg.get("lower", cfg["init"]))
        cfg["init"] = min(cfg["init"], cfg.get("upper", cfg["init"]))

    if cfg.get("lower", 0) >= cfg.get("upper", float("inf")):
        logger.warning(f"  Gradient safety: {name} lower >= upper after clamping, "
                       f"resetting to wide defaults")
        cfg["lower"] = cfg.get("lower", 0) * 0.5
        cfg["upper"] = cfg.get("upper", 1) * 2.0
        if "init" in cfg:
            cfg["init"] = (cfg["lower"] + cfg["upper"]) / 2

    return cfg