"""
Stage 3: Held-Out Validation — Cross-Stimulus Generalization Test
=================================================================

Implements Section 3.4 of the NASS proposal:
    "Run the fitted model forward on held-out stimuli (noise injection,
     ramp current, short square pulses) and compare predicted voltage
     traces to actual recorded traces."

This is the bridge between "the model fits one sweep well" and "the
model generalizes" — the actual scientific claim.

Architecture:
    1. Load the best SGA proposal (channels + fitted params + geometry)
    2. Reconstruct the Jaxley cell with those channels and parameters
    3. Load all held-out sweeps from the Allen train/held-out split
    4. For each held-out sweep:
       a. Resample stimulus to Jaxley timestep (dt=0.025 ms)
       b. Run forward simulation (no gradient, no optimization)
       c. Evaluate against recorded trace using evaluation.py metrics
    5. Produce structured report: per-sweep metrics + aggregates + JSON

Inputs:
    - SGA history JSON (sga_history.json) or a ModelProposal dict
    - Allen data directory with sweep_index.json and NWB files
    - Specimen ID

Outputs:
    - Console summary table
    - JSON report (held_out_report_{specimen_id}.json)
    - Per-sweep CSV (held_out_metrics_{specimen_id}.csv)
    - Optional: overlay plots (simulated vs recorded) saved as PNG

Usage:
    # After SGA run completes:
    python held_out_validation.py --data-dir ./data --specimen-id 509683388

    # With explicit SGA history file:
    python held_out_validation.py --data-dir ./data --specimen-id 509683388 \\
        --sga-history ./data/sga_history.json

    # Called programmatically from run_sga.py:
    from held_out_validation import run_held_out_validation
    report = run_held_out_validation(specimen_id, data_dir, best_proposal)

Requires:
    pip install numpy jax jaxley allensdk matplotlib
"""

# ---- JAX config before any JAX imports ----
from jax import config
config.update("jax_enable_x64", True)

import json
import csv
import argparse
import logging
import time
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, List

import numpy as np
import jax.numpy as jnp
import jaxley as jx
from jaxley.channels import Na, K, Leak

from allensdk.core.cell_types_cache import CellTypesCache

# NASS imports
from channels import NaCortical, Kv3, IM, IAHP, IT, ICaL, IH, CHANNEL_REGISTRY
from evaluation import (
    evaluate_traces,
    EvalResult,
    load_held_out_sweeps,
    window_to_stimulus,
    detect_spikes,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ===========================================================================
# Channel Resolution (mirrors general_fit.py)
# ===========================================================================

BUILTIN_CHANNELS = {"Na": NaCortical, "NaCortical": NaCortical, "K": K, "Leak": Leak}
CUSTOM_CHANNELS = {name: info["class"] for name, info in CHANNEL_REGISTRY.items()}
ALL_CHANNELS = {**BUILTIN_CHANNELS, **CUSTOM_CHANNELS}


# ===========================================================================
# 1. Load Best Proposal from SGA Output
# ===========================================================================

def load_best_proposal(data_dir: Path, sga_history_path: Path = None) -> dict:
    """
    Load the best proposal from SGA history.

    The SGA outer loop saves sga_history.json with all proposals.
    We find the one with the lowest loss that has fitted_params.

    Returns a dict with keys:
        channels, fitted_params, radius, length, capacitance,
        param_config, loss, diagnostics, proposal_id
    """
    if sga_history_path is None:
        sga_history_path = data_dir / "sga_history.json"

    if not sga_history_path.exists():
        raise FileNotFoundError(
            f"SGA history not found at {sga_history_path}. "
            f"Run the SGA outer loop first (python run_sga.py)."
        )

    with open(sga_history_path) as f:
        history = json.load(f)

    # History is a list of proposal dicts from each iteration
    # Find the best one (lowest loss) that has fitted parameters
    best = None
    best_loss = float("inf")

    proposals = history if isinstance(history, list) else history.get("proposals", [])

    for entry in proposals:
        # Handle both direct proposal dicts and nested structures
        proposal = entry.get("proposal", entry)
        loss = proposal.get("loss", float("inf"))
        fitted = proposal.get("fitted_params", {})

        if fitted and loss < best_loss and not np.isnan(loss) and not np.isinf(loss):
            best = proposal
            best_loss = loss

    if best is None:
        raise ValueError(
            "No valid proposals with fitted parameters found in SGA history. "
            "The SGA outer loop may not have converged."
        )

    logger.info(
        f"Loaded best proposal #{best.get('proposal_id', '?')} "
        f"(loss={best_loss:.4f}, channels={best.get('channels', [])})"
    )
    return best


# ===========================================================================
# 2. Reconstruct Jaxley Cell from Proposal
# ===========================================================================

def reconstruct_cell(proposal: dict, dt: float = 0.025) -> jx.Compartment:
    """
    Build a Jaxley single-compartment cell matching the SGA proposal.

    This reproduces exactly what general_fit.py does during training,
    but sets the FITTED parameters (not initial guesses) so the model
    is ready for forward simulation without further optimization.

    Args:
        proposal: dict with 'channels', 'fitted_params', 'radius',
                  'length', 'capacitance'
        dt: simulation timestep in ms

    Returns:
        Configured jx.Compartment ready for simulation
    """
    channels = proposal.get("channels", ["Na", "K", "Leak"])
    fitted = proposal.get("fitted_params", {})

    # Ensure required channels are present
    for required in ["Na", "K", "Leak"]:
        if required not in channels:
            # Check if NaCortical is used instead of Na
            if required == "Na" and "NaCortical" in channels:
                continue
            channels.insert(0, required)
            logger.info(f"  Auto-added required channel: {required}")

    # Build compartment
    comp = jx.Compartment()

    for ch_name in channels:
        if ch_name not in ALL_CHANNELS:
            logger.warning(f"  Unknown channel '{ch_name}' — skipping")
            continue
        comp.insert(ALL_CHANNELS[ch_name]())

    # Set geometry — use fitted values if available, else proposal defaults
    radius = fitted.get("radius", proposal.get("radius", 10.0))
    length = proposal.get("length", 31.4)
    capacitance = fitted.get("capacitance", proposal.get("capacitance", 1.0))

    comp.set("radius", radius)
    comp.set("length", length)
    comp.set("capacitance", capacitance)
    comp.set("axial_resistivity", 100.0)

    # Set reversal potentials
    eNa = fitted.get("eNa", 50.0)
    eK = fitted.get("eK", -90.0)
    comp.set("eNa", eNa)
    comp.set("eK", eK)

    # Set all fitted conductance/parameter values
    for param_name, value in fitted.items():
        # Skip geometry params already set above
        if param_name in ("radius", "length", "capacitance", "eNa", "eK"):
            continue
        try:
            comp.set(param_name, value)
        except Exception as e:
            logger.warning(f"  Could not set {param_name}={value}: {e}")

    logger.info(
        f"  Reconstructed cell: channels={channels}, "
        f"radius={radius:.1f}, length={length:.1f}, "
        f"capacitance={capacitance:.2f}"
    )
    return comp


# ===========================================================================
# 3. Forward Simulation on Held-Out Stimuli
# ===========================================================================

def resample_stimulus(stimulus_nA: np.ndarray, sr_original: float,
                      dt_target_ms: float = 0.025) -> np.ndarray:
    """
    Resample an Allen stimulus waveform to Jaxley's simulation timestep.

    Allen recordings are typically at 200 kHz (dt=0.005 ms).
    Jaxley uses dt=0.025 ms by default.

    Args:
        stimulus_nA: stimulus in nA at original sampling rate
        sr_original: original sampling rate in Hz
        dt_target_ms: target timestep in ms

    Returns:
        Resampled stimulus array in nA
    """
    dt_original_ms = 1000.0 / sr_original
    ratio = dt_target_ms / dt_original_ms

    if abs(ratio - 1.0) < 0.01:
        return stimulus_nA  # already at target rate

    if ratio > 1.0:
        # Downsample: take every Nth sample
        step = max(1, int(round(ratio)))
        return stimulus_nA[::step]
    else:
        # Upsample: interpolate
        n_target = int(len(stimulus_nA) / ratio)
        x_original = np.arange(len(stimulus_nA))
        x_target = np.linspace(0, len(stimulus_nA) - 1, n_target)
        return np.interp(x_target, x_original, stimulus_nA)


def simulate_held_out(cell: jx.Compartment, stimulus_nA: np.ndarray,
                      dt: float = 0.025) -> np.ndarray:
    """
    Run a forward simulation on a held-out stimulus.

    No gradient computation, no optimization — just inject the stimulus
    and record the voltage response.

    Args:
        cell: configured Jaxley compartment with fitted parameters
        stimulus_nA: stimulus waveform in nA at Jaxley timestep
        dt: simulation timestep in ms

    Returns:
        Simulated membrane voltage trace in mV
    """
    # Clear any previous stimuli and recordings
    cell.delete_stimuli()
    cell.delete_recordings()

    # Inject stimulus
    i_ext = jnp.array(stimulus_nA)
    cell.stimulate(i_ext)
    cell.record("v")

    # Run simulation
    v = jx.integrate(cell, delta_t=dt)
    return np.array(v[0])  # shape: (n_timesteps,)


# ===========================================================================
# 4. Full Validation Pipeline
# ===========================================================================

@dataclass
class HeldOutReport:
    """Complete held-out validation report for one specimen."""
    specimen_id: int
    proposal_id: int
    channels: list
    training_loss: float
    n_fitted_params: int
    results: List[Dict] = field(default_factory=list)  # list of EvalResult dicts
    aggregates: Dict = field(default_factory=dict)
    timestamp: str = ""
    wall_time_s: float = 0.0


def run_held_out_validation(
    specimen_id: int,
    data_dir: str,
    proposal: dict = None,
    sga_history_path: str = None,
    dt: float = 0.025,
    max_sweeps_per_category: int = 3,
    save_plots: bool = True,
) -> HeldOutReport:
    """
    Stage 3: Run the best fitted model on all held-out stimuli and evaluate.

    This is the main entry point, callable from run_sga.py or standalone.

    Args:
        specimen_id: Allen specimen ID
        data_dir: path to Allen data cache (with sweep_index.json, NWBs)
        proposal: dict with channels + fitted_params (if None, loads from SGA history)
        sga_history_path: explicit path to sga_history.json
        dt: simulation timestep in ms
        max_sweeps_per_category: limit sweeps evaluated per held-out category
        save_plots: whether to save overlay trace plots

    Returns:
        HeldOutReport with per-sweep metrics and aggregates
    """
    t_start = time.time()
    data_dir = Path(data_dir)
    output_dir = data_dir / "eval"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load proposal ----
    if proposal is None:
        history_path = Path(sga_history_path) if sga_history_path else None
        proposal = load_best_proposal(data_dir, history_path)

    channels = proposal.get("channels", ["Na", "K", "Leak"])
    fitted_params = proposal.get("fitted_params", {})
    training_loss = proposal.get("loss", float("inf"))
    proposal_id = proposal.get("proposal_id", -1)

    print("\n" + "=" * 70)
    print("STAGE 3: HELD-OUT VALIDATION — CROSS-STIMULUS GENERALIZATION")
    print("=" * 70)
    print(f"  Specimen:       {specimen_id}")
    print(f"  Proposal:       #{proposal_id}")
    print(f"  Channels:       {channels}")
    print(f"  Training loss:  {training_loss:.4f}")
    print(f"  Fitted params:  {len(fitted_params)}")
    for k, v in fitted_params.items():
        print(f"    {k}: {v:.6f}")
    print("=" * 70)

    # ---- Reconstruct cell ----
    logger.info("Reconstructing Jaxley cell from fitted parameters...")
    cell = reconstruct_cell(proposal, dt=dt)

    # ---- Load held-out sweeps ----
    logger.info("Loading held-out sweeps from Allen database...")
    ctc = CellTypesCache(manifest_file=str(data_dir / "manifest.json"))

    with open(data_dir / "sweep_index.json") as f:
        sweep_index = json.load(f)

    held_out = load_held_out_sweeps(ctc, specimen_id, sweep_index)

    if not held_out:
        logger.error("No held-out sweeps found! Check sweep_index.json.")
        return HeldOutReport(
            specimen_id=specimen_id, proposal_id=proposal_id,
            channels=channels, training_loss=training_loss,
            n_fitted_params=len(fitted_params),
        )

    n_total = sum(len(sweeps) for sweeps in held_out.values())
    logger.info(
        f"  Found {n_total} held-out sweeps across "
        f"{len(held_out)} categories: {list(held_out.keys())}"
    )

    # ---- Evaluate each held-out sweep ----
    all_results = []

    for category, sweeps in held_out.items():
        logger.info(f"\n  --- {category.upper()} ({len(sweeps)} sweeps) ---")

        for i, sw in enumerate(sweeps[:max_sweeps_per_category]):
            sweep_number = sw["sweep_number"]
            stimulus_nA = sw["stimulus_nA"]
            response_mV = sw["response_mV"]
            sr = sw["sampling_rate"]
            dt_raw = sw["dt_ms"]

            logger.info(
                f"    Sweep {sweep_number}: "
                f"{len(stimulus_nA)} samples at {sr:.0f} Hz"
            )

            try:
                # Window to stimulus region
                stim_win, resp_win = window_to_stimulus(
                    stimulus_nA, response_mV, dt_raw
                )

                # Resample to Jaxley timestep
                stim_resampled = resample_stimulus(stim_win, sr, dt)
                resp_resampled = resample_stimulus(resp_win, sr, dt)

                # Forward simulation
                v_sim = simulate_held_out(cell, stim_resampled, dt=dt)

                # Evaluate
                result = evaluate_traces(
                    target_v=resp_resampled,
                    sim_v=v_sim,
                    dt_ms=dt,
                    specimen_id=specimen_id,
                    model_name=f"SGA-{proposal_id}",
                    sweep_number=sweep_number,
                    stimulus_type=category,
                    n_params=len(fitted_params),
                )

                all_results.append(result)
                logger.info(f"    {result.summary_line()}")

                # Save overlay plot
                if save_plots:
                    _save_overlay_plot(
                        resp_resampled, v_sim, dt, result,
                        category, sweep_number, output_dir
                    )

            except Exception as e:
                logger.error(
                    f"    FAILED on sweep {sweep_number}: {e}",
                    exc_info=True,
                )
                # Record a failure result
                all_results.append(EvalResult(
                    specimen_id=specimen_id,
                    model_name=f"SGA-{proposal_id}",
                    sweep_number=sweep_number,
                    stimulus_type=category,
                    spike_coincidence=0.0,
                    firing_rate_error=1.0,
                    full_trace_mse=float("inf"),
                    model_complexity=len(fitted_params),
                ))

    # ---- Compute aggregates ----
    aggregates = _compute_aggregates(all_results)

    wall_time = time.time() - t_start

    report = HeldOutReport(
        specimen_id=specimen_id,
        proposal_id=proposal_id,
        channels=channels,
        training_loss=training_loss,
        n_fitted_params=len(fitted_params),
        results=[asdict(r) for r in all_results],
        aggregates=aggregates,
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        wall_time_s=wall_time,
    )

    # ---- Print summary ----
    _print_summary(report, all_results)

    # ---- Save outputs ----
    _save_report(report, all_results, output_dir, specimen_id)

    return report


# ===========================================================================
# 5. Aggregate Metrics
# ===========================================================================

def _compute_aggregates(results: List[EvalResult]) -> dict:
    """
    Compute aggregate metrics across all held-out sweeps.

    Groups by stimulus category and computes mean/std for each metric.
    Also computes overall (all-category) aggregates.
    """
    if not results:
        return {}

    # Group by category
    by_category = {}
    for r in results:
        cat = r.stimulus_type
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append(r)

    agg = {}

    # Per-category aggregates
    for cat, cat_results in by_category.items():
        valid = [r for r in cat_results if not np.isinf(r.full_trace_mse)]
        if not valid:
            continue

        gamma_vals = [r.spike_coincidence for r in valid]
        fr_err_vals = [r.firing_rate_error for r in valid]
        r2_vals = [r.subthreshold_r2 for r in valid]
        mse_vals = [r.full_trace_mse for r in valid]

        agg[cat] = {
            "n_sweeps": len(valid),
            "spike_coincidence_mean": float(np.mean(gamma_vals)),
            "spike_coincidence_std": float(np.std(gamma_vals)),
            "firing_rate_error_mean": float(np.mean(fr_err_vals)),
            "firing_rate_error_std": float(np.std(fr_err_vals)),
            "subthreshold_r2_mean": float(np.mean(r2_vals)),
            "subthreshold_r2_std": float(np.std(r2_vals)),
            "full_trace_mse_mean": float(np.mean(mse_vals)),
            "full_trace_mse_std": float(np.std(mse_vals)),
            "total_target_spikes": sum(r.n_target_spikes for r in valid),
            "total_sim_spikes": sum(r.n_sim_spikes for r in valid),
        }

    # Overall aggregates
    all_valid = [r for r in results if not np.isinf(r.full_trace_mse)]
    if all_valid:
        agg["overall"] = {
            "n_sweeps": len(all_valid),
            "n_categories": len(by_category),
            "spike_coincidence_mean": float(np.mean(
                [r.spike_coincidence for r in all_valid]
            )),
            "firing_rate_error_mean": float(np.mean(
                [r.firing_rate_error for r in all_valid]
            )),
            "subthreshold_r2_mean": float(np.mean(
                [r.subthreshold_r2 for r in all_valid]
            )),
            "full_trace_mse_mean": float(np.mean(
                [r.full_trace_mse for r in all_valid]
            )),
        }

    return agg


# ===========================================================================
# 6. Output: Summary, JSON, CSV, Plots
# ===========================================================================

def _print_summary(report: HeldOutReport, results: List[EvalResult]):
    """Print a formatted summary table to console."""
    print(f"\n{'=' * 90}")
    print(f"HELD-OUT VALIDATION RESULTS — Specimen {report.specimen_id}")
    print(f"{'=' * 90}")
    print(f"  Model:    SGA Proposal #{report.proposal_id}")
    print(f"  Channels: {report.channels}")
    print(f"  Training loss: {report.training_loss:.4f}")
    print(f"  Wall time: {report.wall_time_s:.1f}s")
    print()

    # Per-sweep table
    header = (
        f"  {'Stimulus':<25s} {'Sweep':>6s} {'Γ':>6s} {'FR_err':>8s} "
        f"{'R²':>6s} {'Spk(sim)':>9s} {'Spk(tgt)':>9s} {'MSE':>10s}"
    )
    print(header)
    print(f"  {'-' * 82}")

    for r in results:
        mse_str = f"{r.full_trace_mse:.1f}" if not np.isinf(r.full_trace_mse) else "FAILED"
        print(
            f"  {r.stimulus_type:<25s} {r.sweep_number:>6d} "
            f"{r.spike_coincidence:6.3f} {r.firing_rate_error:8.3f} "
            f"{r.subthreshold_r2:6.3f} {r.n_sim_spikes:>9d} "
            f"{r.n_target_spikes:>9d} {mse_str:>10s}"
        )

    # Category aggregates
    if report.aggregates:
        print(f"\n  {'--- CATEGORY AVERAGES ---':^82s}")
        print(f"  {'Category':<25s} {'N':>4s} {'Γ_mean':>8s} {'FR_err':>8s} "
              f"{'R²_mean':>8s} {'MSE_mean':>10s}")
        print(f"  {'-' * 66}")

        for cat, agg in report.aggregates.items():
            if cat == "overall":
                continue
            print(
                f"  {cat:<25s} {agg['n_sweeps']:>4d} "
                f"{agg['spike_coincidence_mean']:8.3f} "
                f"{agg['firing_rate_error_mean']:8.3f} "
                f"{agg['subthreshold_r2_mean']:8.3f} "
                f"{agg['full_trace_mse_mean']:10.1f}"
            )

        if "overall" in report.aggregates:
            ov = report.aggregates["overall"]
            print(f"  {'-' * 66}")
            print(
                f"  {'OVERALL':<25s} {ov['n_sweeps']:>4d} "
                f"{ov['spike_coincidence_mean']:8.3f} "
                f"{ov['firing_rate_error_mean']:8.3f} "
                f"{ov['subthreshold_r2_mean']:8.3f} "
                f"{ov['full_trace_mse_mean']:10.1f}"
            )

    print(f"\n{'=' * 90}\n")


def _save_report(report: HeldOutReport, results: List[EvalResult],
                 output_dir: Path, specimen_id: int):
    """Save JSON report and CSV metrics."""

    # JSON report
    json_path = output_dir / f"held_out_report_{specimen_id}.json"
    with open(json_path, "w") as f:
        json.dump(asdict(report), f, indent=2, default=str)
    logger.info(f"  JSON report saved to {json_path}")

    # CSV metrics
    if results:
        csv_path = output_dir / f"held_out_metrics_{specimen_id}.csv"
        fieldnames = list(asdict(results[0]).keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in results:
                writer.writerow(asdict(r))
        logger.info(f"  CSV metrics saved to {csv_path}")


def _save_overlay_plot(target_v: np.ndarray, sim_v: np.ndarray,
                       dt_ms: float, result: EvalResult,
                       category: str, sweep_number: int,
                       output_dir: Path):
    """Save a voltage trace overlay plot (target vs simulated)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        n = min(len(target_v), len(sim_v))
        t = np.arange(n) * dt_ms

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(t, target_v[:n], "k", linewidth=0.8, alpha=0.7, label="Recorded")
        ax.plot(t, sim_v[:n], "r", linewidth=0.8, alpha=0.7, label="Simulated")
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Voltage (mV)")
        ax.set_title(
            f"Held-Out: {category} — Sweep {sweep_number}\n"
            f"Γ={result.spike_coincidence:.3f}  "
            f"FR_err={result.firing_rate_error:.3f}  "
            f"R²={result.subthreshold_r2:.3f}  "
            f"spk={result.n_sim_spikes}/{result.n_target_spikes}"
        )
        ax.legend(loc="upper right")
        ax.set_xlim(0, t[-1] if len(t) > 0 else 1)

        plot_path = output_dir / f"held_out_{category}_sw{sweep_number}.png"
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.debug(f"    Plot saved: {plot_path}")

    except ImportError:
        logger.debug("    matplotlib not available — skipping plot")
    except Exception as e:
        logger.warning(f"    Plot failed: {e}")


# ===========================================================================
# 7. Integration with run_sga.py
# ===========================================================================

def validate_after_sga(best_proposal, specimen_id: int, data_dir: str,
                       **kwargs) -> Optional[HeldOutReport]:
    """
    Convenience wrapper for calling from run_sga.py after the outer loop.

    Converts a ModelProposal dataclass to a dict and runs validation.

    Args:
        best_proposal: ModelProposal dataclass from sga.py
        specimen_id: Allen specimen ID
        data_dir: path to data directory

    Returns:
        HeldOutReport or None on failure
    """
    # Convert ModelProposal to dict
    if hasattr(best_proposal, "__dict__"):
        proposal_dict = {
            "proposal_id": best_proposal.proposal_id,
            "channels": list(best_proposal.channels),
            "fitted_params": dict(best_proposal.fitted_params),
            "radius": best_proposal.radius,
            "length": best_proposal.length,
            "capacitance": best_proposal.capacitance,
            "param_config": dict(best_proposal.param_config),
            "loss": best_proposal.loss,
            "diagnostics": dict(best_proposal.diagnostics)
                          if best_proposal.diagnostics else {},
        }
    else:
        proposal_dict = best_proposal

    try:
        return run_held_out_validation(
            specimen_id=specimen_id,
            data_dir=data_dir,
            proposal=proposal_dict,
            **kwargs,
        )
    except Exception as e:
        logger.error(f"Held-out validation failed: {e}", exc_info=True)
        return None


# ===========================================================================
# CLI
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Stage 3: Held-out validation — cross-stimulus generalization"
    )
    parser.add_argument(
        "--data-dir", type=str, default="data",
        help="Path to Allen data cache (default: data)"
    )
    parser.add_argument(
        "--specimen-id", type=int, required=True,
        help="Allen specimen ID to validate"
    )
    parser.add_argument(
        "--sga-history", type=str, default=None,
        help="Path to sga_history.json (default: {data-dir}/sga_history.json)"
    )
    parser.add_argument(
        "--dt", type=float, default=0.025,
        help="Simulation timestep in ms (default: 0.025)"
    )
    parser.add_argument(
        "--max-sweeps", type=int, default=3,
        help="Max sweeps to evaluate per held-out category (default: 3)"
    )
    parser.add_argument(
        "--no-plots", action="store_true",
        help="Skip saving overlay trace plots"
    )

    args = parser.parse_args()

    report = run_held_out_validation(
        specimen_id=args.specimen_id,
        data_dir=args.data_dir,
        sga_history_path=args.sga_history,
        dt=args.dt,
        max_sweeps_per_category=args.max_sweeps,
        save_plots=not args.no_plots,
    )

    # Exit code based on whether validation produced any results
    if report.results:
        overall = report.aggregates.get("overall", {})
        gamma = overall.get("spike_coincidence_mean", 0)
        print(f"Validation complete. Overall Γ = {gamma:.3f}")
    else:
        print("Validation produced no results.")
        exit(1)


if __name__ == "__main__":
    main()