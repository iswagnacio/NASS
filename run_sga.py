"""
run_sga.py — Launch the SGA outer loop end-to-end
==================================================

Usage:
    # Basic run with Anthropic API
    python run_sga.py --api-key sk-ant-... --data-dir ./cell_types_data

    # Specify a specimen
    python run_sga.py --api-key sk-ant-... --specimen-id 509683388

    # Use OpenAI instead
    python run_sga.py --api-key sk-... --provider openai --model gpt-4o

    # More iterations, fewer inner epochs (faster but less precise)
    python run_sga.py --api-key sk-ant-... --iterations 8 --inner-epochs 150

    # Use environment variable for API key
    export ANTHROPIC_API_KEY=sk-ant-...
    python run_sga.py
"""

import os
os.environ['JAX_TRACEBACK_FILTERING_MODE'] = 'off'

# ---------------------------------------------------------------------------
# JAX persistent compilation cache
# ---------------------------------------------------------------------------
# Saves compiled XLA kernels between runs. Cold-start runs still pay the
# full compile cost; every subsequent run reuses cached kernels as long as
# the shape/dtype/op graph is unchanged. Typical savings: 30-60 s per cold
# start when iterating on code.
#
# Disable by setting NASS_DISABLE_JIT_CACHE=1 (useful when a JAX version
# bump invalidates cached entries and you want a clean rebuild).
if not os.environ.get("NASS_DISABLE_JIT_CACHE"):
    import jax
    _cache_dir = os.environ.get("NASS_JIT_CACHE_DIR", "./.jax_cache")
    jax.config.update("jax_compilation_cache_dir", _cache_dir)
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)

import json
import logging
import argparse
from pathlib import Path
from dataclasses import asdict
from validation import validate_after_sga

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not installed, rely on env vars or --api-key

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Run the SGA outer loop: LLM proposes → Jaxley fits → diagnose → revise"
    )
    parser.add_argument(
        "--api-key", type=str, default=None,
        help="API key (Anthropic or OpenAI). Falls back to ANTHROPIC_API_KEY "
             "or OPENAI_API_KEY env vars."
    )
    parser.add_argument(
        "--provider", type=str, default="deepseek",
        choices=["anthropic", "openai", "deepseek"],
        help="LLM provider (default: deepseek)"
    )
    parser.add_argument(
        "--model", type=str, default="deepseek-chat",
        help="Model name (default: deepseek-chat)"
    )
    parser.add_argument(
        "--data-dir", type=str, default="cell_types_data",
        help="Path to Allen data cache (default: cell_types_data)"
    )
    parser.add_argument(
        "--specimen-id", type=int, default=None,
        help="Specimen ID. If not set, uses first valid cell."
    )
    parser.add_argument(
        "--iterations", type=int, default=5,
        help="Max outer loop iterations (default: 5)"
    )
    parser.add_argument(
        "--inner-epochs", type=int, default=300,
        help="Gradient descent epochs per inner loop (default: 300)"
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
        "--n-sweeps", type=int, default=1,
        help="Number of training sweeps to fit simultaneously (1=single, 2-3=multi)"
    )
    parser.add_argument(
        "--n-starts", type=int, default=5,
        help="Multi-start probe count per proposal (default: 5). "
             "Set to 1 to skip probing entirely (fastest, use for dev loops); "
             "2-3 gives some diversity at lower cost; 5 is the full search."
    )
    parser.add_argument(
        "--max-duration-ms", type=float, default=1200.0,
        help="Maximum stimulus-window length in ms (default: 1200). "
             "Shorter windows reduce ODE step count linearly — useful for "
             "smoke tests. Do not reduce below the length of your longest "
             "real stimulus for production runs."
    )

    args = parser.parse_args()

    # ---- Resolve API key ----
    api_key = args.api_key
    if api_key is None:
        if args.provider == "anthropic":
            api_key = os.environ.get("ANTHROPIC_API_KEY")
        elif args.provider == "deepseek":
            api_key = os.environ.get("DEEPSEEK_API_KEY")
        else:
            api_key = os.environ.get("OPENAI_API_KEY")

    if not api_key:
        print("ERROR: No API key provided.")
        print("  Options:")
        print("    1. Add ANTHROPIC_API_KEY=sk-ant-... to .env file")
        print("    2. Use --api-key sk-ant-...")
        print("    3. Set ANTHROPIC_API_KEY / OPENAI_API_KEY / DEEPSEEK_API_KEY env var")
        exit(1)

    # ---- Resolve specimen ID ----
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"ERROR: Data directory not found: {data_dir}")
        print("  Run allen_downloader.py first.")
        exit(1)

    specimen_id = args.specimen_id
    if specimen_id is None:
        with open(data_dir / "sweep_index.json") as f:
            sweep_index = json.load(f)
        valid = [int(sid) for sid, e in sweep_index.items() if e.get("valid")]
        if not valid:
            print("ERROR: No valid cells found in sweep_index.json")
            exit(1)
        specimen_id = valid[0]
        logger.info(f"No specimen specified, using first valid: {specimen_id}")

    # ---- Load ephys features if available ----
    ephys_features = {"note": "Features not loaded — using defaults"}
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
                    if not (isinstance(v, float) and v != v)  # skip NaN
                }
                logger.info(f"Loaded {len(ephys_features)} ephys features for {specimen_id}")
        except Exception as e:
            logger.warning(f"Could not load ephys features: {e}")

    # ---- Load neuron metadata ----
    neuron_metadata = {
        "cell_type": "PV+ fast-spiking interneuron",
        "transgenic_line": "Pvalb-IRES-Cre",
        "dendrite_type": "aspiny",
        "brain_region": "VISp",
        "specimen_id": specimen_id,
    }

    # Try to enrich from sweep_index
    try:
        with open(data_dir / "sweep_index.json") as f:
            sweep_index = json.load(f)
        entry = sweep_index.get(str(specimen_id), {})
        if "cortical_layer" in entry:
            neuron_metadata["cortical_layer"] = entry["cortical_layer"]
    except Exception:
        pass

    # ---- Print configuration ----
    print("\n" + "=" * 60)
    print("SGA OUTER LOOP — CONFIGURATION")
    print("=" * 60)
    print(f"  Specimen:       {specimen_id}")
    print(f"  Data dir:       {data_dir}")
    print(f"  Provider:       {args.provider}")
    print(f"  Model:          {args.model}")
    print(f"  Iterations:     {args.iterations}")
    print(f"  Inner epochs:   {args.inner_epochs}")
    print(f"  Inner LR:       {args.inner_lr}")
    print(f"  Top-K:          {args.top_k}")
    print(f"  N sweeps:       {args.n_sweeps}")
    print(f"  N starts:       {args.n_starts}")
    print(f"  Max duration:   {args.max_duration_ms:.1f} ms")
    print("=" * 60 + "\n")

    # ---- Run the outer loop ----
    from sga import OuterLoop

    loop = OuterLoop(
        specimen_id=specimen_id,
        data_dir=str(data_dir),
        api_key=api_key,
        model=args.model,
        provider=args.provider,
        top_k=args.top_k,
        inner_epochs=args.inner_epochs,
        inner_lr=args.inner_lr,
        n_sweeps=args.n_sweeps,
        n_starts=args.n_starts,
        max_duration_ms=args.max_duration_ms,
    )

    best = loop.run(
        max_iterations=args.iterations,
        neuron_metadata=neuron_metadata,
        ephys_features=ephys_features,
    )

    # ---- Print results ----
    print("\n" + "=" * 60)
    print("SGA OUTER LOOP — RESULTS")
    print("=" * 60)

    if best is None:
        print("  No valid proposals found.")
        return

    print(f"  Best proposal: #{best.proposal_id}")
    print(f"  Channels:      {best.channels}")
    print(f"  Loss:          {best.loss:.4f}")
    print(f"  Rationale:     {best.rationale[:300]}")

    if best.fitted_params:
        print(f"\n  Fitted Parameters:")
        for k, v in best.fitted_params.items():
            print(f"    {k}: {v:.6f}")

    if best.diagnostics:
        d = best.diagnostics
        print(f"\n  Diagnostics:")
        print(f"    Spikes:  {d.get('n_sim_spikes', '?')} sim / "
              f"{d.get('n_target_spikes', '?')} target")
        print(f"    r:       {d.get('pearson_r', '?')}")
        if d.get('parameters_at_bounds'):
            print(f"    Bounds:  {d['parameters_at_bounds']}")

    print(f"\n  All proposals in heap:")
    for p in loop.heap.top_k():
        print(f"    #{p.proposal_id}: {p.channels} → loss={p.loss:.2f}")
    
    # ---- Stage 3: Held-out validation ----
    if best is not None and best.fitted_params:
        print("\\n" + "=" * 60)
        print("STAGE 3: HELD-OUT VALIDATION")
        print("=" * 60)
        try:
            report = validate_after_sga(
                best_proposal=best,
                specimen_id=specimen_id,
                data_dir=str(data_dir),
                save_plots=True,
            )
            if report and report.aggregates.get("overall"):
                ov = report.aggregates["overall"]
                print(f"\\n  Overall held-out Γ:      {ov['spike_coincidence_mean']:.3f}")
                print(f"  Overall held-out FR_err: {ov['firing_rate_error_mean']:.3f}")
                print(f"  Overall held-out R²:     {ov['subthreshold_r2_mean']:.3f}")
                print(f"  Overall held-out MSE:    {ov['full_trace_mse_mean']:.1f}")
            else:
                print("  No held-out results (missing sweeps or validation failed)")
        except ImportError:
            print("  validation.py not found — skipping Stage 3")
        except Exception as e:
            print(f"  Stage 3 failed: {e}")
    else:
        print("\\n  Skipping Stage 3: no valid fitted model to validate")

    print(f"\n  History saved to: {data_dir / 'sga_history.json'}")
    print("=" * 60)


if __name__ == "__main__":
    main()