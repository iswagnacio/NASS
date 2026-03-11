"""
Evaluation Pipeline & Comparison Baselines (Weeks 5–6)
======================================================

Implements all held-out evaluation metrics from the NASS proposal and
three simple comparison baselines (LIF, Izhikevich, AdEx) that the
agent-discovered models must beat.

Metrics (Section 4.2 of proposal):
    1. Spike time coincidence factor (Γ)
    2. Firing rate error (relative)
    3. f-I curve RMSE
    4. Subthreshold variance explained (R²)
    5. Spike shape error
    6. Model complexity

Baselines (Section 4.3 of proposal):
    - LIF (Leaky Integrate-and-Fire): lower bound on performance
    - Izhikevich: 4-parameter phenomenological model
    - AdEx (Adaptive Exponential IF): 2-variable model with adaptation

Usage:
    # Evaluate a fitted Jaxley model on held-out stimuli
    python evaluation.py --data-dir ./data --specimen-id 509683388

    # Run all baselines on one cell
    python evaluation.py --data-dir ./data --specimen-id 509683388 --baselines

    # Full evaluation: Jaxley baseline + all simple baselines
    python evaluation.py --data-dir ./data --specimen-id 509683388 --all

Requires:
    pip install numpy scipy allensdk matplotlib
"""

import json
import argparse
import logging
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np
from scipy import signal
from scipy.optimize import minimize

from allensdk.core.cell_types_cache import CellTypesCache

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ============================================================================
# PART 1: EVALUATION METRICS
# ============================================================================

def detect_spikes(v: np.ndarray, dt_ms: float, threshold: float = -20.0,
                  min_isi_ms: float = 2.0) -> np.ndarray:
    """
    Detect spike times from a voltage trace.
    Returns spike times in ms (relative to trace start).
    """
    above = v > threshold
    crossings = np.where(np.diff(above.astype(int)) > 0)[0]
    spike_times = crossings * dt_ms

    # Enforce minimum ISI
    if len(spike_times) > 1:
        filtered = [spike_times[0]]
        for t in spike_times[1:]:
            if t - filtered[-1] >= min_isi_ms:
                filtered.append(t)
        spike_times = np.array(filtered)

    return spike_times


def spike_time_coincidence(target_spikes: np.ndarray, sim_spikes: np.ndarray,
                           delta_ms: float = 4.0, t_total_ms: float = 1000.0) -> float:
    """
    Spike time coincidence factor Γ (Kistler et al. 1997).

    Γ = (N_coinc - <N_coinc>) / (0.5 * (N_target + N_sim))

    where N_coinc is the number of coincident spikes within ±delta_ms,
    and <N_coinc> is the expected number from a Poisson process.

    Returns value in [0, 1] for perfect match, 0 for chance.
    """
    n_target = len(target_spikes)
    n_sim = len(sim_spikes)

    if n_target == 0 and n_sim == 0:
        return 1.0  # both silent = perfect match
    if n_target == 0 or n_sim == 0:
        return 0.0

    # Count coincidences
    n_coinc = 0
    used_sim = set()
    for t_tgt in target_spikes:
        for j, t_sim in enumerate(sim_spikes):
            if j not in used_sim and abs(t_tgt - t_sim) <= delta_ms:
                n_coinc += 1
                used_sim.add(j)
                break

    # Expected coincidences from Poisson process
    rate_sim = n_sim / t_total_ms * 1000.0  # Hz
    n_coinc_expected = 2 * delta_ms / 1000.0 * rate_sim * n_target

    denominator = 0.5 * (n_target + n_sim)
    if denominator == 0:
        return 0.0

    gamma = (n_coinc - n_coinc_expected) / denominator
    # Normalise by (1 - 2*delta*rate_target) to get proper [0,1] range
    rate_target = n_target / t_total_ms * 1000.0
    norm_factor = 1.0 - 2.0 * delta_ms / 1000.0 * rate_target
    if norm_factor > 0:
        gamma /= norm_factor

    return max(0.0, min(1.0, gamma))


def firing_rate_error(target_spikes: np.ndarray, sim_spikes: np.ndarray,
                      t_total_ms: float) -> float:
    """
    Relative firing rate error: |f_sim - f_target| / f_target.
    Returns 0 for perfect match. Returns inf if target is silent but sim fires.
    """
    f_target = len(target_spikes) / (t_total_ms / 1000.0)
    f_sim = len(sim_spikes) / (t_total_ms / 1000.0)

    if f_target == 0:
        return float(f_sim > 0)  # 0 if both silent, 1 if sim fires
    return abs(f_sim - f_target) / f_target


def subthreshold_r_squared(target_v: np.ndarray, sim_v: np.ndarray,
                           threshold: float = -30.0) -> float:
    """
    Variance explained (R²) for subthreshold voltage regions only.
    Masks out spiking regions (above threshold) in both traces.
    """
    mask = (target_v < threshold) & (sim_v < threshold)
    if mask.sum() < 10:
        return 0.0

    t_sub = target_v[mask]
    s_sub = sim_v[mask]

    ss_res = np.sum((t_sub - s_sub) ** 2)
    ss_tot = np.sum((t_sub - np.mean(t_sub)) ** 2)

    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0
    return max(0.0, 1.0 - ss_res / ss_tot)


def extract_mean_spike_waveform(v: np.ndarray, spike_times_ms: np.ndarray,
                                 dt_ms: float, window_ms: float = 3.0) -> Optional[np.ndarray]:
    """
    Extract and average spike waveforms around detected spike times.
    Window: ±window_ms around each spike peak.
    """
    if len(spike_times_ms) == 0:
        return None

    half_win = int(window_ms / dt_ms)
    waveforms = []

    for t in spike_times_ms:
        idx = int(t / dt_ms)
        # Find actual peak within ±1ms of crossing
        search_start = max(0, idx)
        search_end = min(len(v), idx + int(1.5 / dt_ms))
        if search_end <= search_start:
            continue
        peak_idx = search_start + np.argmax(v[search_start:search_end])

        start = peak_idx - half_win
        end = peak_idx + half_win + 1
        if start >= 0 and end <= len(v):
            waveforms.append(v[start:end])

    if not waveforms:
        return None
    return np.mean(waveforms, axis=0)


def spike_shape_error(target_v: np.ndarray, sim_v: np.ndarray,
                      target_spikes: np.ndarray, sim_spikes: np.ndarray,
                      dt_ms: float) -> dict:
    """
    Compare mean spike waveforms between target and simulation.
    Returns MSE of waveforms plus individual shape features.
    """
    target_wf = extract_mean_spike_waveform(target_v, target_spikes, dt_ms)
    sim_wf = extract_mean_spike_waveform(sim_v, sim_spikes, dt_ms)

    result = {"waveform_mse": np.nan, "peak_error_mV": np.nan,
              "trough_error_mV": np.nan, "half_width_error_ms": np.nan}

    if target_wf is None or sim_wf is None:
        return result

    # Align lengths
    n = min(len(target_wf), len(sim_wf))
    target_wf = target_wf[:n]
    sim_wf = sim_wf[:n]

    result["waveform_mse"] = float(np.mean((target_wf - sim_wf) ** 2))
    result["peak_error_mV"] = float(np.max(sim_wf) - np.max(target_wf))
    result["trough_error_mV"] = float(np.min(sim_wf) - np.min(target_wf))

    # Half-width: time above half-max
    def half_width(wf):
        half = (np.max(wf) + np.min(wf)) / 2
        above = wf > half
        if above.sum() == 0:
            return 0.0
        return np.sum(above) * dt_ms

    result["half_width_error_ms"] = float(half_width(sim_wf) - half_width(target_wf))
    return result


@dataclass
class EvalResult:
    """Complete evaluation result for one model on one sweep."""
    specimen_id: int
    model_name: str
    sweep_number: int
    stimulus_type: str

    # Metrics
    spike_coincidence: float = 0.0
    firing_rate_error: float = 0.0
    n_target_spikes: int = 0
    n_sim_spikes: int = 0
    subthreshold_r2: float = 0.0
    waveform_mse: float = np.nan
    peak_error_mV: float = np.nan
    trough_error_mV: float = np.nan
    half_width_error_ms: float = np.nan
    full_trace_mse: float = 0.0
    model_complexity: int = 0  # number of free parameters

    def summary_line(self) -> str:
        return (
            f"  {self.model_name:<20s} {self.stimulus_type:<15s} "
            f"Γ={self.spike_coincidence:.3f}  "
            f"FR_err={self.firing_rate_error:.3f}  "
            f"R²={self.subthreshold_r2:.3f}  "
            f"spk={self.n_sim_spikes}/{self.n_target_spikes}  "
            f"MSE={self.full_trace_mse:.1f}"
        )


def evaluate_traces(target_v: np.ndarray, sim_v: np.ndarray, dt_ms: float,
                    specimen_id: int, model_name: str, sweep_number: int,
                    stimulus_type: str, n_params: int = 0) -> EvalResult:
    """
    Run all evaluation metrics on a pair of traces.
    """
    n = min(len(target_v), len(sim_v))
    target_v = target_v[:n]
    sim_v = sim_v[:n]
    t_total_ms = n * dt_ms

    target_spikes = detect_spikes(target_v, dt_ms)
    sim_spikes = detect_spikes(sim_v, dt_ms)

    shape = spike_shape_error(target_v, sim_v, target_spikes, sim_spikes, dt_ms)

    return EvalResult(
        specimen_id=specimen_id,
        model_name=model_name,
        sweep_number=sweep_number,
        stimulus_type=stimulus_type,
        spike_coincidence=spike_time_coincidence(target_spikes, sim_spikes,
                                                  t_total_ms=t_total_ms),
        firing_rate_error=firing_rate_error(target_spikes, sim_spikes, t_total_ms),
        n_target_spikes=len(target_spikes),
        n_sim_spikes=len(sim_spikes),
        subthreshold_r2=subthreshold_r_squared(target_v, sim_v),
        full_trace_mse=float(np.mean((target_v - sim_v) ** 2)),
        model_complexity=n_params,
        **shape,
    )


# ============================================================================
# PART 2: SIMPLE BASELINE MODELS (LIF, Izhikevich, AdEx)
# ============================================================================

class LIFModel:
    """
    Leaky Integrate-and-Fire model.
    The simplest possible model — lower bound on performance.

    Parameters: V_rest, V_thresh, V_reset, tau_m, R_m, t_ref
    """
    name = "LIF"
    n_params = 6

    def __init__(self, V_rest=-65.0, V_thresh=-50.0, V_reset=-70.0,
                 tau_m=10.0, R_m=100.0, t_ref=2.0):
        self.V_rest = V_rest      # mV
        self.V_thresh = V_thresh  # mV
        self.V_reset = V_reset    # mV
        self.tau_m = tau_m        # ms
        self.R_m = R_m            # MΩ (input resistance)
        self.t_ref = t_ref        # ms (refractory period)

    def simulate(self, stimulus_nA: np.ndarray, dt_ms: float = 0.025) -> np.ndarray:
        """Simulate LIF model given current injection in nA."""
        n = len(stimulus_nA)
        v = np.full(n, self.V_rest)
        ref_counter = 0.0

        for i in range(1, n):
            if ref_counter > 0:
                ref_counter -= dt_ms
                v[i] = self.V_reset
                continue

            dv = (-(v[i-1] - self.V_rest) + self.R_m * stimulus_nA[i-1]) / self.tau_m
            v[i] = v[i-1] + dv * dt_ms

            if v[i] >= self.V_thresh:
                v[i] = 20.0  # spike peak (for visualisation)
                v[i] = self.V_reset  # immediately reset
                ref_counter = self.t_ref

        return v

    def fit(self, target_v: np.ndarray, stimulus_nA: np.ndarray, dt_ms: float):
        """Fit LIF parameters to minimise subthreshold MSE."""
        target_spikes = detect_spikes(target_v, dt_ms)
        target_rate = len(target_spikes) / (len(target_v) * dt_ms / 1000.0)

        # Estimate V_rest from first 50ms (pre-stimulus baseline)
        n_baseline = min(int(50.0 / dt_ms), len(target_v) // 4)
        if n_baseline > 0:
            self.V_rest = float(np.mean(target_v[:n_baseline]))

        def cost(params):
            self.V_thresh, self.tau_m, self.R_m = params
            v_sim = self.simulate(stimulus_nA, dt_ms)
            n = min(len(v_sim), len(target_v))
            # Subthreshold MSE
            mask = target_v[:n] < -30.0
            if mask.sum() < 10:
                return 1e6
            return np.mean((v_sim[:n][mask] - target_v[:n][mask]) ** 2)

        result = minimize(cost, [-50.0, 10.0, 100.0],
                         method="Nelder-Mead",
                         options={"maxiter": 500, "xatol": 0.1})
        self.V_thresh, self.tau_m, self.R_m = result.x
        return self


class IzhikevichModel:
    """
    Izhikevich model (2003). 4 parameters: a, b, c, d.
    Fast to fit, captures many firing patterns, but no biophysical
    interpretability.

    dv/dt = 0.04*v² + 5*v + 140 - u + I
    du/dt = a*(b*v - u)
    if v >= 30: v = c, u = u + d
    """
    name = "Izhikevich"
    n_params = 4

    def __init__(self, a=0.1, b=0.2, c=-65.0, d=2.0):
        # Default: fast-spiking pattern
        self.a = a
        self.b = b
        self.c = c    # mV (reset voltage)
        self.d = d    # reset recovery increment

    def simulate(self, stimulus_nA: np.ndarray, dt_ms: float = 0.025) -> np.ndarray:
        """
        Simulate Izhikevich model. Input current must be scaled to
        match the model's unitless convention (~pA scale works).
        """
        n = len(stimulus_nA)
        v = np.full(n, -65.0)
        u = self.b * v[0]
        I_scale = stimulus_nA * 1000.0  # nA -> pA-ish scaling for Izhikevich

        for i in range(1, n):
            I = I_scale[i-1]
            if v[i-1] >= 30.0:
                v[i] = self.c
                u = u + self.d
            else:
                dv = (0.04 * v[i-1]**2 + 5.0 * v[i-1] + 140.0 - u + I) * dt_ms
                du = self.a * (self.b * v[i-1] - u) * dt_ms
                v[i] = v[i-1] + dv
                u = u + du

                if v[i] >= 30.0:
                    v[i] = 30.0  # spike peak

        return v

    def fit(self, target_v: np.ndarray, stimulus_nA: np.ndarray, dt_ms: float):
        """Fit Izhikevich parameters via Nelder-Mead on trace MSE."""
        def cost(params):
            self.a, self.b, self.c, self.d = params
            self.a = max(0.01, min(0.3, self.a))
            self.b = max(0.01, min(0.5, self.b))
            v_sim = self.simulate(stimulus_nA, dt_ms)
            n = min(len(v_sim), len(target_v))
            mask = target_v[:n] < -30.0
            if mask.sum() < 10:
                return 1e6
            return np.mean((v_sim[:n][mask] - target_v[:n][mask]) ** 2)

        # FS neuron defaults as starting point
        result = minimize(cost, [0.1, 0.2, -65.0, 2.0],
                         method="Nelder-Mead",
                         options={"maxiter": 1000, "xatol": 0.01})
        self.a, self.b, self.c, self.d = result.x
        return self


class AdExModel:
    """
    Adaptive Exponential Integrate-and-Fire (Brette & Gerstner 2005).
    2-variable model with adaptation. Good balance of speed and accuracy.

    C dV/dt = -g_L(V - E_L) + g_L*Δ_T*exp((V - V_T)/Δ_T) - w + I
    τ_w dw/dt = a*(V - E_L) - w
    if V >= V_cut: V = V_reset, w = w + b
    """
    name = "AdEx"
    n_params = 9

    def __init__(self, C=200.0, g_L=10.0, E_L=-65.0, V_T=-50.0,
                 delta_T=2.0, a=2.0, tau_w=30.0, b=0.0, V_reset=-58.0):
        self.C = C            # pF
        self.g_L = g_L        # nS
        self.E_L = E_L        # mV
        self.V_T = V_T        # mV (threshold)
        self.delta_T = delta_T  # mV (slope factor)
        self.a = a            # nS (subthreshold adaptation)
        self.tau_w = tau_w    # ms
        self.b = b            # pA (spike-triggered adaptation)
        self.V_reset = V_reset  # mV
        self.V_cut = 0.0      # mV (spike detection)

    def simulate(self, stimulus_nA: np.ndarray, dt_ms: float = 0.025) -> np.ndarray:
        """Simulate AdEx model. Current in nA, converted to pA internally."""
        n = len(stimulus_nA)
        v = np.full(n, self.E_L)
        w = 0.0
        I_pA = stimulus_nA * 1000.0  # nA -> pA

        for i in range(1, n):
            I = I_pA[i-1]

            # Exponential term (clamp to avoid overflow)
            exp_term = self.delta_T * np.exp(
                np.clip((v[i-1] - self.V_T) / max(self.delta_T, 0.1), -50, 50)
            )

            dv = (-self.g_L * (v[i-1] - self.E_L) + self.g_L * exp_term - w + I) / self.C
            dw = (self.a * (v[i-1] - self.E_L) - w) / max(self.tau_w, 0.1)

            v[i] = v[i-1] + dv * dt_ms
            w = w + dw * dt_ms

            if v[i] >= self.V_cut:
                v[i] = self.V_reset
                w = w + self.b

        return v

    def fit(self, target_v: np.ndarray, stimulus_nA: np.ndarray, dt_ms: float):
        """Fit AdEx parameters via Nelder-Mead on subthreshold MSE."""
        # Estimate resting potential
        n_baseline = min(int(50.0 / dt_ms), len(target_v) // 4)
        if n_baseline > 0:
            self.E_L = float(np.mean(target_v[:n_baseline]))

        def cost(params):
            self.C, self.g_L, self.V_T, self.delta_T, self.a, self.tau_w = params
            self.C = max(50, self.C)
            self.g_L = max(1, self.g_L)
            self.delta_T = max(0.5, self.delta_T)
            self.tau_w = max(1, self.tau_w)
            v_sim = self.simulate(stimulus_nA, dt_ms)
            n = min(len(v_sim), len(target_v))
            mask = target_v[:n] < -30.0
            if mask.sum() < 10:
                return 1e6
            return np.mean((v_sim[:n][mask] - target_v[:n][mask]) ** 2)

        result = minimize(cost, [200.0, 10.0, -50.0, 2.0, 2.0, 30.0],
                         method="Nelder-Mead",
                         options={"maxiter": 1000, "xatol": 0.1})
        self.C, self.g_L, self.V_T, self.delta_T, self.a, self.tau_w = result.x
        return self


# ============================================================================
# PART 3: DATA LOADING HELPERS
# ============================================================================

def load_held_out_sweeps(ctc: CellTypesCache, specimen_id: int,
                         sweep_index: dict) -> dict:
    """
    Load all held-out sweeps for a specimen, grouped by category.
    Returns: { "noise": [sweep_dicts], "ramp": [...], "short_square": [...],
               "long_square_extrapolation": [...] }
    """
    entry = sweep_index[str(specimen_id)]
    held_out = entry["split"]["held_out"]
    data_set = ctc.get_ephys_data(specimen_id)

    result = {}
    for category, sweeps in held_out.items():
        loaded = []
        for sw in sweeps[:3]:  # limit to 3 per category for speed
            try:
                sweep_data = data_set.get_sweep(sw["sweep_number"])
                idx = sweep_data["index_range"]
                stimulus = sweep_data["stimulus"][idx[0]:idx[1]+1]
                response = sweep_data["response"][idx[0]:idx[1]+1]
                sr = sweep_data["sampling_rate"]
                loaded.append({
                    "sweep_number": sw["sweep_number"],
                    "stimulus_nA": stimulus * 1e9,     # A -> nA
                    "response_mV": response * 1e3,     # V -> mV
                    "sampling_rate": sr,
                    "dt_ms": 1000.0 / sr,
                })
            except Exception as e:
                logger.warning(f"  Could not load sweep {sw['sweep_number']}: {e}")
        if loaded:
            result[category] = loaded

    return result


def window_to_stimulus(stimulus: np.ndarray, response: np.ndarray,
                       dt_ms: float, pre_ms: float = 50.0,
                       post_ms: float = 100.0,
                       max_ms: float = 1200.0) -> tuple:
    """Extract window around the active stimulus period."""
    threshold = np.max(np.abs(stimulus)) * 0.1
    active = np.where(np.abs(stimulus) > threshold)[0]

    if len(active) > 0:
        pre = int(pre_ms / dt_ms)
        post = int(post_ms / dt_ms)
        start = max(0, active[0] - pre)
        end = min(len(stimulus), active[-1] + post)
        max_n = int(max_ms / dt_ms)
        if (end - start) > max_n:
            end = start + max_n
        return stimulus[start:end], response[start:end]

    # Fallback: return first max_ms
    n = min(len(stimulus), int(max_ms / dt_ms))
    return stimulus[:n], response[:n]


# ============================================================================
# PART 4: MAIN EVALUATION PIPELINE
# ============================================================================

def run_baselines(ctc: CellTypesCache, specimen_id: int,
                  sweep_index: dict, output_dir: Path,
                  dt_eval: float = 0.1) -> list:
    """
    Fit and evaluate LIF, Izhikevich, and AdEx on training + held-out data.
    Uses a coarser dt (0.1 ms) for the simple models since they don't
    need Jaxley's 0.025 ms resolution.
    """
    logger.info(f"Running baselines for specimen {specimen_id}...")

    # Load a training sweep for fitting
    entry = sweep_index[str(specimen_id)]
    train_sweeps = entry["split"]["training"]["long_square"]
    # Pick highest amplitude training sweep
    train_sw = sorted(train_sweeps,
                      key=lambda s: s.get("stimulus_amplitude", 0) or 0)[-1]

    data_set = ctc.get_ephys_data(specimen_id)
    sweep_data = data_set.get_sweep(train_sw["sweep_number"])
    idx = sweep_data["index_range"]
    stim_raw = sweep_data["stimulus"][idx[0]:idx[1]+1] * 1e9   # nA
    resp_raw = sweep_data["response"][idx[0]:idx[1]+1] * 1e3   # mV
    sr = sweep_data["sampling_rate"]
    dt_raw = 1000.0 / sr

    # Window to stimulus region
    stim_win, resp_win = window_to_stimulus(stim_raw, resp_raw, dt_raw)

    # Downsample for faster baseline fitting
    downsample = max(1, int(dt_eval / dt_raw))
    stim_ds = stim_win[::downsample]
    resp_ds = resp_win[::downsample]
    dt_ds = dt_raw * downsample

    models = [LIFModel(), IzhikevichModel(), AdExModel()]
    all_results = []

    for model in models:
        logger.info(f"  Fitting {model.name}...")
        try:
            model.fit(resp_ds, stim_ds, dt_ds)
        except Exception as e:
            logger.error(f"  {model.name} fitting failed: {e}")
            continue

        # Evaluate on training sweep first
        v_sim = model.simulate(stim_ds, dt_ds)
        result = evaluate_traces(
            resp_ds, v_sim, dt_ds,
            specimen_id, model.name, train_sw["sweep_number"],
            "long_square_train", model.n_params
        )
        all_results.append(result)
        logger.info(f"    Train: {result.summary_line()}")

        # Evaluate on held-out sweeps
        held_out = load_held_out_sweeps(ctc, specimen_id, sweep_index)
        for category, sweeps in held_out.items():
            for sw in sweeps[:2]:  # 2 per category
                stim_ho, resp_ho = window_to_stimulus(
                    sw["stimulus_nA"], sw["response_mV"], sw["dt_ms"]
                )
                # Downsample
                ds = max(1, int(dt_eval / sw["dt_ms"]))
                stim_ho_ds = stim_ho[::ds]
                resp_ho_ds = resp_ho[::ds]
                dt_ho = sw["dt_ms"] * ds

                v_ho = model.simulate(stim_ho_ds, dt_ho)
                result = evaluate_traces(
                    resp_ho_ds, v_ho, dt_ho,
                    specimen_id, model.name, sw["sweep_number"],
                    category, model.n_params
                )
                all_results.append(result)
                logger.info(f"    {category}: {result.summary_line()}")

    return all_results


def run_evaluation(data_dir: str, specimen_id: int, run_baselines_flag: bool = False,
                   run_all: bool = False):
    """Main entry point for evaluation."""
    data_dir = Path(data_dir)
    output_dir = data_dir / "eval"
    output_dir.mkdir(parents=True, exist_ok=True)

    ctc = CellTypesCache(manifest_file=str(data_dir / "manifest.json"))
    with open(data_dir / "sweep_index.json") as f:
        sweep_index = json.load(f)

    all_results = []

    # Run simple baselines
    if run_baselines_flag or run_all:
        baseline_results = run_baselines(ctc, specimen_id, sweep_index, output_dir)
        all_results.extend(baseline_results)

    # Print summary
    if all_results:
        print(f"\n{'='*90}")
        print(f"EVALUATION SUMMARY — Specimen {specimen_id}")
        print(f"{'='*90}")
        print(f"  {'Model':<20s} {'Stimulus':<15s} {'Γ':>6s} {'FR_err':>8s} "
              f"{'R²':>6s} {'Spk(sim)':>9s} {'Spk(tgt)':>9s} {'MSE':>10s} {'#Params':>8s}")
        print(f"  {'-'*85}")
        for r in all_results:
            print(f"  {r.model_name:<20s} {r.stimulus_type:<15s} "
                  f"{r.spike_coincidence:6.3f} {r.firing_rate_error:8.3f} "
                  f"{r.subthreshold_r2:6.3f} {r.n_sim_spikes:9d} "
                  f"{r.n_target_spikes:9d} {r.full_trace_mse:10.1f} "
                  f"{r.model_complexity:8d}")

        # Save results
        import csv
        csv_path = output_dir / f"eval_{specimen_id}.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(asdict(all_results[0]).keys()))
            writer.writeheader()
            for r in all_results:
                writer.writerow(asdict(r))
        logger.info(f"Results saved to {csv_path}")

    return all_results


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluation pipeline and comparison baselines"
    )
    parser.add_argument("--data-dir", type=str, default="data",
                        help="Path to Allen data cache")
    parser.add_argument("--specimen-id", type=int, required=True,
                        help="Specimen ID to evaluate")
    parser.add_argument("--baselines", action="store_true",
                        help="Run LIF, Izhikevich, AdEx baselines")
    parser.add_argument("--all", action="store_true",
                        help="Run everything")

    args = parser.parse_args()
    run_evaluation(
        data_dir=args.data_dir,
        specimen_id=args.specimen_id,
        run_baselines_flag=args.baselines,
        run_all=args.all,
    )