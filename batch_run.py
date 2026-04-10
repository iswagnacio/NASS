"""
Cross-Neuron Generalization — Batch Pipeline Runner
====================================================

Implements proposal Sections 4.4 and months 7–8:
    "Run the full pipeline on 5–10 more PV+ specimens with working sweep
     selection. This tests whether the system reliably converges, not just
     on the one specimen you've been debugging with."

Two modes of operation:

1. FULL PIPELINE (default):
   Run SGA Stage 1→2→3 independently on each specimen. Tests whether the
   system reliably converges across diverse PV+ neurons.

2. TRANSFER TEST (--transfer-from):
   Take the channel structure discovered on a reference specimen and run
   only Stage 2 inner loop (parameter re-fitting) on other specimens.
   Tests whether the agent discovers a true type-level model, not just
   an individual-neuron overfit (proposal Section 4.4).

Outputs:
    - Per-specimen: sga_history.json, held_out_report, trace plots
    - Cross-neuron: batch_summary.json with convergence rates, aggregate
      metrics, failure analysis
    - Console: formatted comparison table

Usage:
    # Full pipeline on 10 PV+ neurons
    python cross_neuron.py --data-dir ./cell_types_data --n-specimens 10

    # Transfer test: use 509683388's channel structure on other neurons
    python cross_neuron.py --data-dir ./cell_types_data --n-specimens 10 \\
        --transfer-from 509683388

    # Specify exact specimen IDs
    python cross_neuron.py --data-dir ./cell_types_data \\
        --specimen-ids 509683388 486755781 479704527

    # Dry run: just list which specimens would be tested
    python cross_neuron.py --data-dir ./cell_types_data --n-specimens 10 --dry-run

Requires:
    pip install numpy jax jaxley allensdk
    + ANTHROPIC_API_KEY or OPENAI_API_KEY env var (for full pipeline mode)
"""

import os

import json
import logging
import argparse
import time
import traceback
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict

import numpy as np

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ===========================================================================
# 1. Specimen Selection
# ===========================================================================

def select_specimens(data_dir: Path, n_specimens: int = 10,
                     specimen_ids: List[int] = None,
                     exclude_ids: List[int] = None) -> List[int]:
    """
    Select PV+ specimens for cross-neuron testing.

    Selection criteria (in order of priority):
      1. Must be valid in sweep_index.json (enough training + held-out sweeps)
      2. Must have at least 1 spiking training sweep
      3. Prefer specimens with more held-out categories available
      4. Prefer specimens with more training sweeps (more amplitudes)

    Args:
        data_dir: path to Allen data cache
        n_specimens: how many specimens to select
        specimen_ids: explicit list (overrides automatic selection)
        exclude_ids: IDs to skip (e.g., the reference specimen in transfer mode)

    Returns:
        List of specimen IDs
    """
    if specimen_ids:
        return specimen_ids

    exclude_ids = set(exclude_ids or [])

    with open(data_dir / "sweep_index.json") as f:
        sweep_index = json.load(f)

    candidates = []
    for sid_str, entry in sweep_index.items():
        sid = int(sid_str)
        if sid in exclude_ids:
            continue
        if not entry.get("valid", False):
            continue

        summary = entry.get("split", {}).get("summary", {})
        n_train = summary.get("n_train_long_square", 0)
        n_heldout_cats = sum(1 for k in ["n_noise", "n_ramp", "n_short_square"]
                            if summary.get(k, 0) > 0)

        # Check for spiking sweeps
        train_sweeps = entry.get("split", {}).get("training", {}).get("long_square", [])
        has_spiking = any((sw.get("num_spikes") or 0) > 0 for sw in train_sweeps)

        if n_train < 3:
            continue

        candidates.append({
            "specimen_id": sid,
            "n_train": n_train,
            "n_heldout_cats": n_heldout_cats,
            "has_spiking": has_spiking,
            # Composite score: prioritise spiking + more held-out + more train
            "score": (10 if has_spiking else 0) + n_heldout_cats * 3 + n_train,
        })

    # Sort by score descending
    candidates.sort(key=lambda c: c["score"], reverse=True)

    selected = [c["specimen_id"] for c in candidates[:n_specimens]]

    logger.info(
        f"Selected {len(selected)}/{len(candidates)} valid PV+ specimens "
        f"(from {len(sweep_index)} total)"
    )
    return selected


# ===========================================================================
# 2. Per-Specimen Result
# ===========================================================================

@dataclass
class SpecimenResult:
    """Result of running the pipeline on one specimen."""
    specimen_id: int
    mode: str  # "full_pipeline" or "transfer"
    status: str = "pending"  # "success", "failed", "no_spikes", "no_convergence"
    error: str = ""
    wall_time_s: float = 0.0

    # SGA output (Stage 2)
    best_loss: float = float("inf")
    channels: list = field(default_factory=list)
    n_sga_iterations: int = 0
    n_sim_spikes: int = 0
    n_target_spikes: int = 0
    pearson_r: float = 0.0

    # Stage 3 held-out metrics
    held_out_gamma: float = 0.0
    held_out_fr_error: float = 0.0
    held_out_r2: float = 0.0
    held_out_mse: float = 0.0
    n_held_out_sweeps: int = 0


# ===========================================================================
# 3. Full Pipeline Mode — SGA Stage 1→2→3 per specimen
# ===========================================================================

def run_full_pipeline_specimen(
    specimen_id: int,
    data_dir: Path,
    api_key: str,
    provider: str = "anthropic",
    model: str = "claude-sonnet-4-20250514",
    sga_iterations: int = 5,
    inner_epochs: int = 300,
    inner_lr: float = 0.02,
    top_k: int = 5,
) -> SpecimenResult:
    """
    Run the complete SGA pipeline (Stage 1→2→3) on one specimen.

    This is essentially what run_sga.py does, but wrapped to return
    structured results instead of printing to console.
    """
    result = SpecimenResult(
        specimen_id=specimen_id,
        mode="full_pipeline",
    )
    t_start = time.time()

    try:
        # ---- Load metadata ----
        with open(data_dir / "sweep_index.json") as f:
            sweep_index = json.load(f)

        ephys_features = {"note": "default"}
        features_path = data_dir / "pv_ephys_features.csv"
        if features_path.exists():
            try:
                import pandas as pd
                df = pd.read_csv(features_path, index_col=0)
                if specimen_id in df.index:
                    row = df.loc[specimen_id]
                    ephys_features = {
                        k: float(v) if isinstance(v, (int, float)) else str(v)
                        for k, v in row.items()
                        if not (isinstance(v, float) and v != v)
                    }
            except Exception:
                pass

        neuron_metadata = {
            "cell_type": "PV+ fast-spiking interneuron",
            "transgenic_line": "Pvalb-IRES-Cre",
            "dendrite_type": "aspiny",
            "brain_region": "VISp",
            "specimen_id": specimen_id,
        }

        # ---- Run SGA outer loop ----
        from sga import OuterLoop

        loop = OuterLoop(
            specimen_id=specimen_id,
            data_dir=str(data_dir),
            api_key=api_key,
            model=model,
            provider=provider,
            top_k=top_k,
            inner_epochs=inner_epochs,
            inner_lr=inner_lr,
        )

        best = loop.run(
            max_iterations=sga_iterations,
            neuron_metadata=neuron_metadata,
            ephys_features=ephys_features,
        )

        if best is None or not best.fitted_params:
            result.status = "no_convergence"
            result.n_sga_iterations = sga_iterations
            result.wall_time_s = time.time() - t_start
            return result

        # Record Stage 2 results
        result.best_loss = best.loss
        result.channels = list(best.channels)
        result.n_sga_iterations = best.iteration + 1
        diag = best.diagnostics or {}
        result.n_sim_spikes = diag.get("n_sim_spikes", 0)
        result.n_target_spikes = diag.get("n_target_spikes", 0)
        result.pearson_r = diag.get("pearson_r", 0.0)

        # ---- Run Stage 3 held-out validation ----
        from held_out_validation import validate_after_sga
        report = validate_after_sga(
            best_proposal=best,
            specimen_id=specimen_id,
            data_dir=str(data_dir),
            save_plots=True,
        )

        if report and report.aggregates.get("overall"):
            ov = report.aggregates["overall"]
            result.held_out_gamma = ov.get("spike_coincidence_mean", 0)
            result.held_out_fr_error = ov.get("firing_rate_error_mean", 0)
            result.held_out_r2 = ov.get("subthreshold_r2_mean", 0)
            result.held_out_mse = ov.get("full_trace_mse_mean", 0)
            result.n_held_out_sweeps = ov.get("n_sweeps", 0)

        result.status = "success"

    except Exception as e:
        result.status = "failed"
        result.error = f"{type(e).__name__}: {e}"
        logger.error(f"  Specimen {specimen_id} failed: {e}", exc_info=True)

    result.wall_time_s = time.time() - t_start
    return result


# ===========================================================================
# 4. Transfer Mode — Re-fit parameters only (Section 4.4)
# ===========================================================================

def run_transfer_specimen(
    specimen_id: int,
    data_dir: Path,
    reference_proposal: dict,
    inner_epochs: int = 300,
    inner_lr: float = 0.02,
) -> SpecimenResult:
    """
    Transfer test: take a channel structure from a reference neuron and
    re-fit only parameters on this specimen (skip Stage 1).

    This tests whether the agent discovered a true type-level model
    or just an individual-neuron overfit (proposal Section 4.4).
    """
    result = SpecimenResult(
        specimen_id=specimen_id,
        mode="transfer",
        channels=list(reference_proposal.get("channels", [])),
    )
    t_start = time.time()

    try:
        from sga import ModelProposal
        from general_fit import fit_proposal

        # Create a proposal with the reference's channel structure
        # but fresh param_config (will use FALLBACK_PARAM_BOUNDS)
        proposal = ModelProposal(
            proposal_id=0,
            iteration=0,
            channels=list(reference_proposal["channels"]),
            param_config=dict(reference_proposal.get("param_config", {})),
            radius=reference_proposal.get("radius", 10.0),
            length=reference_proposal.get("length", 31.4),
            capacitance=reference_proposal.get("capacitance", 1.0),
            rationale=f"Transfer from specimen {reference_proposal.get('specimen_id', '?')}",
        )

        # Run inner loop only (no LLM, no outer loop)
        report = fit_proposal(
            proposal=proposal,
            specimen_id=specimen_id,
            data_dir=str(data_dir),
            epochs=inner_epochs,
            lr=inner_lr,
        )

        result.best_loss = report.final_loss
        result.n_sim_spikes = report.n_sim_spikes
        result.n_target_spikes = report.n_target_spikes
        result.pearson_r = report.pearson_r
        result.n_sga_iterations = 0  # no SGA, just inner loop

        if report.proposal and report.proposal.fitted_params:
            # Run held-out validation
            from held_out_validation import validate_after_sga
            ho_report = validate_after_sga(
                best_proposal=report.proposal,
                specimen_id=specimen_id,
                data_dir=str(data_dir),
                save_plots=True,
            )

            if ho_report and ho_report.aggregates.get("overall"):
                ov = ho_report.aggregates["overall"]
                result.held_out_gamma = ov.get("spike_coincidence_mean", 0)
                result.held_out_fr_error = ov.get("firing_rate_error_mean", 0)
                result.held_out_r2 = ov.get("subthreshold_r2_mean", 0)
                result.held_out_mse = ov.get("full_trace_mse_mean", 0)
                result.n_held_out_sweeps = ov.get("n_sweeps", 0)

            result.status = "success"
        else:
            result.status = "no_convergence"

    except Exception as e:
        result.status = "failed"
        result.error = f"{type(e).__name__}: {e}"
        logger.error(f"  Transfer specimen {specimen_id} failed: {e}",
                     exc_info=True)

    result.wall_time_s = time.time() - t_start
    return result


# ===========================================================================
# 5. Batch Runner
# ===========================================================================

@dataclass
class BatchSummary:
    """Aggregate results across all specimens."""
    mode: str
    n_specimens: int = 0
    n_success: int = 0
    n_failed: int = 0
    n_no_convergence: int = 0
    total_wall_time_s: float = 0.0
    results: List[Dict] = field(default_factory=list)

    # Aggregate metrics (computed from successful specimens only)
    convergence_rate: float = 0.0
    mean_training_loss: float = 0.0
    mean_held_out_gamma: float = 0.0
    mean_held_out_fr_error: float = 0.0
    mean_held_out_r2: float = 0.0
    mean_held_out_mse: float = 0.0

    # Channel structure analysis (full pipeline mode)
    channel_frequency: Dict = field(default_factory=dict)


def run_batch(
    data_dir: str,
    specimen_ids: List[int] = None,
    n_specimens: int = 10,
    transfer_from: int = None,
    api_key: str = None,
    provider: str = "anthropic",
    model: str = "claude-sonnet-4-20250514",
    sga_iterations: int = 5,
    inner_epochs: int = 300,
    inner_lr: float = 0.02,
    top_k: int = 5,
    dry_run: bool = False,
) -> BatchSummary:
    """
    Run the pipeline across multiple PV+ specimens.

    Args:
        data_dir: path to Allen data cache
        specimen_ids: explicit list of IDs (overrides n_specimens)
        n_specimens: how many to test (ignored if specimen_ids given)
        transfer_from: if set, use this specimen's channel structure for transfer
        api_key: LLM API key (required for full pipeline mode)
        dry_run: just select specimens and print, don't run

    Returns:
        BatchSummary with per-specimen results and aggregates
    """
    data_dir = Path(data_dir)
    t_start = time.time()

    is_transfer = transfer_from is not None
    mode = "transfer" if is_transfer else "full_pipeline"

    # ---- Select specimens ----
    exclude = [transfer_from] if transfer_from else None
    selected = select_specimens(
        data_dir, n_specimens=n_specimens,
        specimen_ids=specimen_ids, exclude_ids=exclude,
    )

    if not selected:
        print("ERROR: No valid specimens found.")
        return BatchSummary(mode=mode)

    print("\n" + "=" * 70)
    print(f"CROSS-NEURON GENERALIZATION — {mode.upper()}")
    print("=" * 70)
    print(f"  Specimens: {len(selected)}")
    if is_transfer:
        print(f"  Transfer from: {transfer_from}")
    print(f"  Specimen IDs: {selected}")
    print("=" * 70)

    if dry_run:
        print("\n  [DRY RUN] Would run pipeline on these specimens.")
        _print_specimen_info(data_dir, selected)
        return BatchSummary(mode=mode, n_specimens=len(selected))

    # ---- Load reference proposal for transfer mode ----
    reference_proposal = None
    if is_transfer:
        from held_out_validation import load_best_proposal
        reference_proposal = load_best_proposal(data_dir)
        reference_proposal["specimen_id"] = transfer_from
        print(f"\n  Reference model: channels={reference_proposal['channels']}")
        print(f"  Reference loss: {reference_proposal.get('loss', '?')}")

    # ---- Validate API key for full pipeline ----
    if not is_transfer:
        if api_key is None:
            if provider == "anthropic":
                api_key = os.environ.get("ANTHROPIC_API_KEY")
            else:
                api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            print("ERROR: Full pipeline mode requires an API key.")
            print("  Set ANTHROPIC_API_KEY env var or use --api-key")
            return BatchSummary(mode=mode, n_specimens=len(selected))

    # ---- Run each specimen ----
    all_results = []

    for i, sid in enumerate(selected):
        print(f"\n{'─' * 70}")
        print(f"  [{i+1}/{len(selected)}] Specimen {sid}")
        print(f"{'─' * 70}")

        if is_transfer:
            result = run_transfer_specimen(
                specimen_id=sid,
                data_dir=data_dir,
                reference_proposal=reference_proposal,
                inner_epochs=inner_epochs,
                inner_lr=inner_lr,
            )
        else:
            result = run_full_pipeline_specimen(
                specimen_id=sid,
                data_dir=data_dir,
                api_key=api_key,
                provider=provider,
                model=model,
                sga_iterations=sga_iterations,
                inner_epochs=inner_epochs,
                inner_lr=inner_lr,
                top_k=top_k,
            )

        all_results.append(result)

        # Print inline status
        status_icon = {"success": "✓", "failed": "✗",
                       "no_convergence": "○", "no_spikes": "○"
                       }.get(result.status, "?")
        print(
            f"  {status_icon} {sid}: {result.status} "
            f"(loss={result.best_loss:.2f}, "
            f"spk={result.n_sim_spikes}/{result.n_target_spikes}, "
            f"Γ_ho={result.held_out_gamma:.3f}, "
            f"time={result.wall_time_s:.0f}s)"
        )

    # ---- Compute summary ----
    summary = _compute_batch_summary(all_results, mode, time.time() - t_start)

    # ---- Print and save ----
    _print_batch_summary(summary)
    _save_batch_summary(summary, data_dir)

    return summary


# ===========================================================================
# 6. Summary Computation
# ===========================================================================

def _compute_batch_summary(results: List[SpecimenResult], mode: str,
                           total_time: float) -> BatchSummary:
    """Compute aggregate metrics from per-specimen results."""
    summary = BatchSummary(
        mode=mode,
        n_specimens=len(results),
        total_wall_time_s=total_time,
        results=[asdict(r) for r in results],
    )

    summary.n_success = sum(1 for r in results if r.status == "success")
    summary.n_failed = sum(1 for r in results if r.status == "failed")
    summary.n_no_convergence = sum(1 for r in results
                                    if r.status == "no_convergence")

    if summary.n_specimens > 0:
        summary.convergence_rate = summary.n_success / summary.n_specimens

    # Metrics from successful runs only
    ok = [r for r in results if r.status == "success"]
    if ok:
        summary.mean_training_loss = float(np.mean([r.best_loss for r in ok]))
        summary.mean_held_out_gamma = float(np.mean(
            [r.held_out_gamma for r in ok]))
        summary.mean_held_out_fr_error = float(np.mean(
            [r.held_out_fr_error for r in ok]))
        summary.mean_held_out_r2 = float(np.mean(
            [r.held_out_r2 for r in ok]))
        summary.mean_held_out_mse = float(np.mean(
            [r.held_out_mse for r in ok]))

    # Channel frequency analysis (full pipeline mode)
    if mode == "full_pipeline":
        ch_counts = {}
        for r in ok:
            for ch in r.channels:
                ch_counts[ch] = ch_counts.get(ch, 0) + 1
        summary.channel_frequency = ch_counts

    return summary


# ===========================================================================
# 7. Output: Print, Save
# ===========================================================================

def _print_specimen_info(data_dir: Path, specimen_ids: List[int]):
    """Print info about selected specimens (for dry run)."""
    with open(data_dir / "sweep_index.json") as f:
        sweep_index = json.load(f)

    print(f"\n  {'ID':<12s} {'Train':<8s} {'Noise':<8s} {'Ramp':<8s} "
          f"{'ShortSq':<10s} {'LS_held':<10s} {'Spiking':<8s}")
    print(f"  {'─' * 64}")

    for sid in specimen_ids:
        entry = sweep_index.get(str(sid), {})
        s = entry.get("split", {}).get("summary", {})
        train = entry.get("split", {}).get("training", {}).get("long_square", [])
        has_spiking = "yes" if any((sw.get("num_spikes") or 0) > 0
                                   for sw in train) else "no"
        print(
            f"  {sid:<12d} {s.get('n_train_long_square', 0):<8d} "
            f"{s.get('n_noise', 0):<8d} {s.get('n_ramp', 0):<8d} "
            f"{s.get('n_short_square', 0):<10d} "
            f"{s.get('n_heldout_long_square', 0):<10d} {has_spiking:<8s}"
        )


def _print_batch_summary(summary: BatchSummary):
    """Print formatted cross-neuron summary."""
    print(f"\n{'=' * 80}")
    print(f"CROSS-NEURON SUMMARY — {summary.mode.upper()}")
    print(f"{'=' * 80}")
    print(f"  Specimens tested:  {summary.n_specimens}")
    print(f"  Converged:         {summary.n_success} "
          f"({summary.convergence_rate:.0%})")
    print(f"  Failed:            {summary.n_failed}")
    print(f"  No convergence:    {summary.n_no_convergence}")
    print(f"  Total wall time:   {summary.total_wall_time_s:.0f}s "
          f"({summary.total_wall_time_s / 60:.1f} min)")

    if summary.n_success > 0:
        print(f"\n  Aggregate Metrics (from {summary.n_success} successful runs):")
        print(f"    Mean training loss:   {summary.mean_training_loss:.2f}")
        print(f"    Mean held-out Γ:      {summary.mean_held_out_gamma:.3f}")
        print(f"    Mean held-out FR_err: {summary.mean_held_out_fr_error:.3f}")
        print(f"    Mean held-out R²:     {summary.mean_held_out_r2:.3f}")
        print(f"    Mean held-out MSE:    {summary.mean_held_out_mse:.1f}")

    if summary.channel_frequency:
        print(f"\n  Channel frequency across successful runs:")
        for ch, count in sorted(summary.channel_frequency.items(),
                                key=lambda x: -x[1]):
            pct = count / summary.n_success * 100
            print(f"    {ch:<15s} {count}/{summary.n_success} ({pct:.0f}%)")

    # Per-specimen table
    print(f"\n  {'ID':<12s} {'Status':<16s} {'Loss':>8s} {'Spk':>10s} "
          f"{'r':>6s} {'Γ_ho':>6s} {'FR_err':>8s} {'R²_ho':>6s} "
          f"{'MSE_ho':>8s} {'Time':>6s}")
    print(f"  {'─' * 96}")

    for r_dict in summary.results:
        sid = r_dict["specimen_id"]
        status = r_dict["status"]
        loss = r_dict["best_loss"]
        spk = f"{r_dict['n_sim_spikes']}/{r_dict['n_target_spikes']}"
        r = r_dict.get("pearson_r", 0)
        gamma = r_dict.get("held_out_gamma", 0)
        fr_err = r_dict.get("held_out_fr_error", 0)
        r2 = r_dict.get("held_out_r2", 0)
        mse = r_dict.get("held_out_mse", 0)
        t = r_dict.get("wall_time_s", 0)

        loss_str = f"{loss:.2f}" if not np.isinf(loss) else "inf"
        print(
            f"  {sid:<12d} {status:<16s} {loss_str:>8s} {spk:>10s} "
            f"{r:6.3f} {gamma:6.3f} {fr_err:8.3f} {r2:6.3f} "
            f"{mse:8.1f} {t:5.0f}s"
        )

        if r_dict.get("error"):
            print(f"    └─ Error: {r_dict['error'][:80]}")

    print(f"\n{'=' * 80}\n")


def _save_batch_summary(summary: BatchSummary, data_dir: Path):
    """Save batch results to JSON."""
    output_dir = data_dir / "eval"
    output_dir.mkdir(parents=True, exist_ok=True)

    path = output_dir / f"batch_summary_{summary.mode}.json"
    with open(path, "w") as f:
        json.dump(asdict(summary), f, indent=2, default=str)
    logger.info(f"Batch summary saved to {path}")


# ===========================================================================
# CLI
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Cross-neuron generalization: batch pipeline runner"
    )
    parser.add_argument(
        "--data-dir", type=str, default="cell_types_data",
        help="Path to Allen data cache (default: cell_types_data)"
    )
    parser.add_argument(
        "--n-specimens", type=int, default=10,
        help="Number of PV+ specimens to test (default: 10)"
    )
    parser.add_argument(
        "--specimen-ids", type=int, nargs="+", default=None,
        help="Explicit specimen IDs (overrides --n-specimens)"
    )
    parser.add_argument(
        "--transfer-from", type=int, default=None,
        help="Transfer mode: use this specimen's channel structure on others"
    )
    parser.add_argument(
        "--api-key", type=str, default=None,
        help="LLM API key (falls back to env vars)"
    )
    parser.add_argument(
        "--provider", type=str, default="anthropic",
        choices=["anthropic", "openai"],
        help="LLM provider (default: anthropic)"
    )
    parser.add_argument(
        "--model", type=str, default="claude-sonnet-4-20250514",
        help="Model name (default: claude-sonnet-4-20250514)"
    )
    parser.add_argument(
        "--sga-iterations", type=int, default=5,
        help="SGA outer loop iterations per specimen (default: 5)"
    )
    parser.add_argument(
        "--inner-epochs", type=int, default=300,
        help="Inner loop epochs (default: 300)"
    )
    parser.add_argument(
        "--inner-lr", type=float, default=0.02,
        help="Inner loop learning rate (default: 0.02)"
    )
    parser.add_argument(
        "--top-k", type=int, default=5,
        help="Top-K heap size (default: 5)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Just list selected specimens, don't run"
    )

    args = parser.parse_args()

    summary = run_batch(
        data_dir=args.data_dir,
        specimen_ids=args.specimen_ids,
        n_specimens=args.n_specimens,
        transfer_from=args.transfer_from,
        api_key=args.api_key,
        provider=args.provider,
        model=args.model,
        sga_iterations=args.sga_iterations,
        inner_epochs=args.inner_epochs,
        inner_lr=args.inner_lr,
        top_k=args.top_k,
        dry_run=args.dry_run,
    )

    # Exit code
    if summary.n_success == 0 and summary.n_specimens > 0:
        exit(1)


if __name__ == "__main__":
    main()