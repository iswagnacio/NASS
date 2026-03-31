"""
Allen Cell Types Database — PV+ Fast-Spiking Interneuron Data Pipeline
======================================================================

Downloads NWB electrophysiology data for Pvalb (PV+) fast-spiking interneurons
from the Allen Cell Types Database, extracts sweeps by stimulus type, splits
into training and held-out sets, and loads precomputed electrophysiology features.

Usage:
    python allen_downloader.py                    # full pipeline
    python allen_downloader.py --list-only        # just list PV+ cells, no download
    python allen_downloader.py --max-cells 5      # download only 5 cells (for testing)
    python allen_downloader.py --cache-dir ./data  # custom cache directory

Output structure:
    {cache_dir}/
    ├── manifest.json
    ├── pv_cells_metadata.csv          # all PV+ cell metadata
    ├── pv_ephys_features.csv          # precomputed electrophysiology features
    ├── sweep_index.json               # per-cell sweep inventory + train/held-out splits
    └── specimen_{id}/
        └── ephys.nwb                  # raw NWB files (managed by Allen SDK cache)
"""

import os
import json
import argparse
import logging
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

from allensdk.core.cell_types_cache import CellTypesCache
from allensdk.api.queries.cell_types_api import CellTypesApi

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Stimulus name mappings — Allen uses these names in sweep metadata.
# Variants exist across different recording rigs; we normalise them here.
LONG_SQUARE_NAMES = {
    "Long Square",
    "Long Square - Triple",
}

NOISE_NAMES = {
    "Noise 1",
    "Noise 2",
}

RAMP_NAMES = {
    "Ramp",
}

SHORT_SQUARE_NAMES = {
    "Short Square",
    "Short Square - Triple",
}

# Stimulus categories for the proposal's train / held-out split
TRAINING_STIMULI = LONG_SQUARE_NAMES
HELDOUT_STIMULI = {
    "noise": NOISE_NAMES,
    "ramp": RAMP_NAMES,
    "short_square": SHORT_SQUARE_NAMES,
    "long_square_heldout": set(),  # populated per-cell: amplitudes not used in training
}

# Transgenic line substrings that mark PV+ (Pvalb) cells
PVALB_LINE_SUBSTRINGS = ["Pvalb"]

# We only want mouse cortical neurons
SPECIES = [CellTypesApi.MOUSE]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Cell Discovery — find PV+ fast-spiking interneurons
# ---------------------------------------------------------------------------

def get_pv_cells(ctc: CellTypesCache) -> pd.DataFrame:
    """
    Query all mouse cells, then filter to those from Pvalb transgenic lines
    with aspiny dendrites (canonical PV+ fast-spiking interneuron markers).

    Returns a DataFrame with one row per cell.
    """
    logger.info("Fetching full cell metadata from Allen Cell Types Database...")
    all_cells = ctc.get_cells(species=SPECIES)
    logger.info(f"  Total mouse cells: {len(all_cells)}")

    pv_cells = []
    for cell in all_cells:
        tg = cell.get("transgenic_line") or ""
        if any(sub in tg for sub in PVALB_LINE_SUBSTRINGS):
            pv_cells.append(cell)

    df = pd.DataFrame(pv_cells)
    logger.info(f"  PV+ (Pvalb-line) cells: {len(df)}")

    # Aspiny dendrites are the canonical PV+ FS morphology, but we keep
    # all Pvalb-line cells and flag dendrite type so users can filter later.
    if "dendrite_type" in df.columns:
        aspiny_count = (df["dendrite_type"] == "aspiny").sum()
        logger.info(f"    of which aspiny: {aspiny_count}")

    return df


# ---------------------------------------------------------------------------
# 2. Sweep Inventory — classify every sweep by stimulus type
# ---------------------------------------------------------------------------

def classify_sweep(stimulus_name: str) -> str:
    """Map an Allen stimulus_name string to one of our canonical categories."""
    if stimulus_name in LONG_SQUARE_NAMES:
        return "long_square"
    if stimulus_name in NOISE_NAMES:
        return "noise"
    if stimulus_name in RAMP_NAMES:
        return "ramp"
    if stimulus_name in SHORT_SQUARE_NAMES:
        return "short_square"
    return "other"


def get_sweep_inventory(ctc: CellTypesCache, specimen_id: int) -> dict:
    """
    Download sweep metadata for one cell and return a structured inventory:
    {
      "specimen_id": int,
      "sweeps_by_category": { "long_square": [...], "noise": [...], ... },
      "all_sweeps": [ { sweep metadata dicts } ]
    }
    """
    sweeps = ctc.get_ephys_sweeps(specimen_id)

    by_category = defaultdict(list)
    for sw in sweeps:
        cat = classify_sweep(sw["stimulus_name"])
        by_category[cat].append({
            "sweep_number": sw["sweep_number"],
            "stimulus_name": sw["stimulus_name"],
            "stimulus_units": sw.get("stimulus_units", ""),
            "stimulus_amplitude": sw.get("stimulus_absolute_amplitude"),
            "num_spikes": sw.get("num_spikes"),
        })

    return {
        "specimen_id": specimen_id,
        "sweeps_by_category": dict(by_category),
        "total_sweeps": len(sweeps),
    }


# ---------------------------------------------------------------------------
# 2b. Robust Spike Detection from Raw Voltage Traces
# ---------------------------------------------------------------------------

def count_spikes_from_trace(response: np.ndarray, sampling_rate: float,
                            stimulus: np.ndarray = None,
                            threshold_mV: float = None) -> dict:
    """
    Count spikes directly from a voltage trace when Allen metadata is missing.

    Uses an adaptive threshold based on Vrest (pre-stimulus baseline).
    If no stimulus array is provided, uses the first 10% of the trace as baseline.

    Returns dict with:
        - num_spikes: int
        - vrest_mV: float (estimated resting potential)
        - threshold_mV: float (threshold used for detection)
        - spike_times_ms: list of spike peak times in ms
    """
    # Convert to mV if values look like they're in volts (< 1.0 range)
    v = np.array(response, dtype=np.float64)
    if np.abs(np.median(v)) < 0.2:  # probably in volts
        v = v * 1e3  # convert to mV

    n_samples = len(v)
    dt_s = 1.0 / sampling_rate
    dt_ms = dt_s * 1e3

    # ---- Determine pre-stimulus baseline for Vrest ----
    if stimulus is not None:
        stim = np.array(stimulus, dtype=np.float64)
        # Find stimulus onset: first index where |stimulus| > 5% of max
        stim_abs = np.abs(stim)
        stim_max = np.max(stim_abs) if np.max(stim_abs) > 0 else 1.0
        onset_mask = stim_abs > 0.05 * stim_max
        if np.any(onset_mask):
            onset_idx = np.argmax(onset_mask)
            # Use 80% of pre-stimulus period, skip first 5% for settling
            skip = max(1, int(onset_idx * 0.05))
            baseline_end = max(skip + 10, int(onset_idx * 0.80))
            baseline_v = v[skip:baseline_end]
        else:
            # No detectable stimulus — use first 10%
            baseline_v = v[:max(10, n_samples // 10)]
    else:
        baseline_v = v[:max(10, n_samples // 10)]

    vrest = float(np.median(baseline_v))

    # ---- Adaptive spike threshold ----
    # For cortical neurons: Vrest is typically -60 to -75 mV,
    # spike threshold is typically -40 to -20 mV.
    # Use midpoint between Vrest and a generous spike peak estimate,
    # but clamp to physiological range.
    if threshold_mV is None:
        # Adaptive: halfway between Vrest and -10 mV (conservative peak estimate)
        threshold_mV = max(-30.0, min(-10.0, (vrest + (-10.0)) / 2.0))
        # But never lower than Vrest + 15 mV (need clear separation from baseline)
        threshold_mV = max(threshold_mV, vrest + 15.0)

    # ---- Detect upward threshold crossings ----
    above = v > threshold_mV
    crossings = np.diff(above.astype(np.int32)) > 0  # False->True transitions
    crossing_indices = np.where(crossings)[0]

    # ---- Refine: find actual peak within each crossing, enforce refractory period ----
    min_refractory_ms = 1.5  # minimum inter-spike interval for cortical neurons
    min_refractory_samples = max(1, int(min_refractory_ms / dt_ms))

    spike_peaks = []
    last_peak_idx = -min_refractory_samples - 1

    for cross_idx in crossing_indices:
        if cross_idx - last_peak_idx < min_refractory_samples:
            continue  # too close to last spike — skip

        # Search for peak within 2 ms after crossing
        search_end = min(n_samples, cross_idx + int(2.0 / dt_ms) + 1)
        peak_idx = cross_idx + np.argmax(v[cross_idx:search_end])

        # Validate: peak must be substantially above threshold
        peak_v = v[peak_idx]
        if peak_v < threshold_mV + 5.0:
            continue  # marginal crossing, not a real spike

        spike_peaks.append(peak_idx)
        last_peak_idx = peak_idx

    spike_times_ms = [float(idx * dt_ms) for idx in spike_peaks]

    return {
        "num_spikes": len(spike_peaks),
        "vrest_mV": vrest,
        "threshold_mV": threshold_mV,
        "spike_times_ms": spike_times_ms,
    }


def enrich_sweep_spike_counts(data_set, sweeps: list,
                              sampling_rate: float = None) -> int:
    """
    For a list of sweep dicts where num_spikes is None, load the raw
    voltage trace and count spikes directly. Mutates the sweep dicts
    in-place (sets 'num_spikes' and 'spike_detection_method').

    Tries three methods in order:
      1. Allen's get_spike_times() — if available and non-empty
      2. Direct voltage trace threshold crossing — always works

    Returns the number of sweeps that were enriched.
    """
    enriched = 0
    for sw in sweeps:
        if sw.get("num_spikes") is not None:
            continue  # already has metadata

        sweep_number = sw["sweep_number"]
        try:
            # Method 1: Try Allen's precomputed spike times
            spike_times = data_set.get_spike_times(sweep_number)
            if len(spike_times) > 0:
                sw["num_spikes"] = int(len(spike_times))
                sw["spike_detection_method"] = "allen_spike_times"
                enriched += 1
                continue
        except Exception:
            pass

        try:
            # Method 2: Count from raw voltage trace
            sweep_data = data_set.get_sweep(sweep_number)
            idx = sweep_data["index_range"]
            response = sweep_data["response"][idx[0]:idx[1]+1]
            stimulus = sweep_data["stimulus"][idx[0]:idx[1]+1]
            sr = sweep_data["sampling_rate"]

            result = count_spikes_from_trace(response, sr, stimulus=stimulus)
            sw["num_spikes"] = result["num_spikes"]
            sw["vrest_mV"] = result["vrest_mV"]
            sw["spike_detection_method"] = "trace_threshold"
            enriched += 1
        except Exception as e:
            logger.warning(f"    Could not count spikes for sweep {sweep_number}: {e}")
            sw["num_spikes"] = 0
            sw["spike_detection_method"] = "failed"
            enriched += 1

    return enriched


# ---------------------------------------------------------------------------
# 3. Train / Held-Out Split
# ---------------------------------------------------------------------------

def make_train_heldout_split(inventory: dict, train_fraction: float = 0.7) -> dict:
    """
    Split sweeps into training and held-out sets following the proposal:

    Training:  Long-square sweeps at a subset of amplitudes (spanning
               sub-threshold to ~2× rheobase).
    Held-out:
        Set 1  — Noise 1 & Noise 2 (all)
        Set 2  — Ramp (all)
        Set 3  — Short Square (all)
        Set 4  — Long-square amplitudes NOT used in training

    For long-square sweeps we sort by amplitude and assign the first
    `train_fraction` of unique amplitudes to training, the rest to held-out
    Set 4.  This tests f-I curve interpolation/extrapolation.
    """
    cats = inventory["sweeps_by_category"]

    # ---- Long square: split by amplitude ----
    ls_sweeps = cats.get("long_square", [])
    amplitudes = sorted(
        set(sw["stimulus_amplitude"] for sw in ls_sweeps
            if sw["stimulus_amplitude"] is not None)
    )

    if amplitudes:
        n_train = max(1, int(len(amplitudes) * train_fraction))
        train_amps = set(amplitudes[:n_train])
        heldout_amps = set(amplitudes[n_train:])
    else:
        train_amps, heldout_amps = set(), set()

    ls_train = [sw for sw in ls_sweeps
                if sw["stimulus_amplitude"] in train_amps]
    ls_heldout = [sw for sw in ls_sweeps
                  if sw["stimulus_amplitude"] in heldout_amps]

    # ---- Other categories: all go to held-out ----
    split = {
        "training": {
            "long_square": ls_train,
        },
        "held_out": {
            "noise": cats.get("noise", []),
            "ramp": cats.get("ramp", []),
            "short_square": cats.get("short_square", []),
            "long_square_extrapolation": ls_heldout,
        },
        "summary": {
            "n_train_long_square": len(ls_train),
            "n_train_amplitudes": len(train_amps),
            "n_heldout_long_square": len(ls_heldout),
            "n_heldout_amplitudes": len(heldout_amps),
            "n_noise": len(cats.get("noise", [])),
            "n_ramp": len(cats.get("ramp", [])),
            "n_short_square": len(cats.get("short_square", [])),
        },
    }
    return split


# ---------------------------------------------------------------------------
# 4. Electrophysiology Features
# ---------------------------------------------------------------------------

def get_ephys_features(ctc: CellTypesCache, pv_ids: list) -> pd.DataFrame:
    """
    Download precomputed electrophysiology features for all cells, then
    filter to our PV+ set.

    Key features for the proposal:
        - vrest               (resting potential, mV)
        - ri                  (input resistance, MΩ)
        - tau                 (membrane time constant, ms)
        - threshold_i_long_square  (rheobase, pA)
        - avg_firing_rate     (Hz, at rheobase)
        - adaptation          (adaptation index)
        - upstroke_downstroke_ratio_long_square
        - f_i_curve_slope     (Hz/pA)
    """
    logger.info("Downloading precomputed electrophysiology features...")
    ef = ctc.get_ephys_features()
    ef_df = pd.DataFrame(ef)
    pv_ef = ef_df[ef_df["specimen_id"].isin(pv_ids)].copy()
    logger.info(f"  Features available for {len(pv_ef)}/{len(pv_ids)} PV+ cells")
    return pv_ef


# ---------------------------------------------------------------------------
# 5. Sweep Data Extraction Helpers
# ---------------------------------------------------------------------------

def extract_sweep(data_set, sweep_number: int) -> dict:
    """
    Pull stimulus, response, sampling rate, and spike times for one sweep.
    Returns numpy arrays in SI units (amps for stimulus, volts for response).
    """
    sweep_data = data_set.get_sweep(sweep_number)
    try:
        spike_times = data_set.get_spike_times(sweep_number)
    except Exception:
        spike_times = np.array([])

    idx = sweep_data["index_range"]
    stimulus = sweep_data["stimulus"][idx[0]:idx[1]+1]
    response = sweep_data["response"][idx[0]:idx[1]+1]
    sampling_rate = sweep_data["sampling_rate"]
    time = np.arange(len(stimulus)) / sampling_rate

    return {
        "time": time,
        "stimulus": stimulus,
        "response": response,
        "sampling_rate": sampling_rate,
        "spike_times": spike_times,
    }


def load_cell_sweeps(ctc: CellTypesCache, specimen_id: int,
                     sweep_numbers: list) -> list:
    """
    Download NWB for a cell (if not cached) and extract specified sweeps.
    Returns a list of dicts from extract_sweep().
    """
    data_set = ctc.get_ephys_data(specimen_id)
    results = []
    for sn in sweep_numbers:
        try:
            results.append({"sweep_number": sn, **extract_sweep(data_set, sn)})
        except Exception as e:
            logger.warning(f"  Could not extract sweep {sn} for {specimen_id}: {e}")
    return results


# ---------------------------------------------------------------------------
# 6. Validation — check a cell has enough data to be useful
# ---------------------------------------------------------------------------

def validate_cell(inventory: dict, split: dict,
                  min_train_sweeps: int = 3,
                  min_heldout_categories: int = 1) -> tuple[bool, str]:
    """
    Check that a cell has enough sweeps to be useful for training and
    evaluation.  Returns (is_valid, reason).
    """
    n_train = split["summary"]["n_train_long_square"]
    if n_train < min_train_sweeps:
        return False, f"only {n_train} training sweeps (need >= {min_train_sweeps})"

    heldout_cats = sum([
        1 for k in ["n_noise", "n_ramp", "n_short_square"]
        if split["summary"][k] > 0
    ])
    if heldout_cats < min_heldout_categories:
        return False, f"only {heldout_cats} held-out categories (need >= {min_heldout_categories})"

    return True, "ok"


# ---------------------------------------------------------------------------
# 7. Main Pipeline
# ---------------------------------------------------------------------------

def run_pipeline(cache_dir: str = "cell_types_data",
                 max_cells: int = None,
                 list_only: bool = False,
                 download_nwb: bool = True):
    """
    End-to-end pipeline:
      1. Find PV+ fast-spiking cells
      2. Download ephys features
      3. For each cell: get sweep inventory, make train/held-out split, validate
      4. Optionally download NWB files and verify sweep extraction
      5. Enrich sweep spike counts from raw traces when metadata is missing
      6. Save all metadata to disk
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    ctc = CellTypesCache(manifest_file=str(cache_dir / "manifest.json"))

    # --- Step 1: Find PV+ cells ---
    pv_df = get_pv_cells(ctc)
    pv_df.to_csv(cache_dir / "pv_cells_metadata.csv", index=False)
    logger.info(f"Saved metadata for {len(pv_df)} PV+ cells")

    if list_only:
        print(f"\n{'='*70}")
        print(f"Found {len(pv_df)} PV+ (Pvalb-line) cells in Allen Cell Types DB")
        print(f"{'='*70}")
        cols = ["id", "name", "transgenic_line", "dendrite_type",
                "structure_layer_name", "structure_area_abbrev"]
        cols = [c for c in cols if c in pv_df.columns]
        print(pv_df[cols].to_string(index=False))
        return

    pv_ids = pv_df["id"].tolist()
    if max_cells:
        pv_ids = pv_ids[:max_cells]
        logger.info(f"Limiting to first {max_cells} cells for testing")

    # --- Step 2: Electrophysiology features ---
    ef_df = get_ephys_features(ctc, pv_ids)
    ef_df.to_csv(cache_dir / "pv_ephys_features.csv", index=False)

    # Print summary of key features
    key_features = [
        "vrest", "ri", "tau", "threshold_i_long_square",
        "avg_firing_rate_long_square", "adaptation",
        "upstroke_downstroke_ratio_long_square",
        "f_i_curve_slope",
    ]
    available = [f for f in key_features if f in ef_df.columns]
    if available:
        print(f"\n{'='*70}")
        print("Electrophysiology Feature Summary (PV+ cells)")
        print(f"{'='*70}")
        print(ef_df[available].describe().round(2).to_string())

    # --- Step 3 & 4: Per-cell sweep inventory, splits, optional NWB download ---
    sweep_index = {}
    valid_cells = []
    skipped_cells = []

    for i, specimen_id in enumerate(pv_ids):
        logger.info(f"[{i+1}/{len(pv_ids)}] Processing specimen {specimen_id}...")

        # Sweep inventory (downloads only sweep metadata JSON, not NWB)
        try:
            inventory = get_sweep_inventory(ctc, specimen_id)
        except Exception as e:
            logger.error(f"  Failed to get sweeps: {e}")
            skipped_cells.append({"id": specimen_id, "reason": str(e)})
            continue

        # --- Step 4b: Enrich spike counts from NWB when metadata is missing ---
        # Check if any long_square sweeps are missing num_spikes
        ls_sweeps = inventory["sweeps_by_category"].get("long_square", [])
        missing_count = sum(1 for sw in ls_sweeps if sw.get("num_spikes") is None)

        if missing_count > 0 and download_nwb:
            logger.info(f"  {missing_count}/{len(ls_sweeps)} long_square sweeps "
                        f"missing num_spikes — scanning raw traces...")
            try:
                data_set = ctc.get_ephys_data(specimen_id)
                n_enriched = enrich_sweep_spike_counts(data_set, ls_sweeps)
                logger.info(f"  Enriched {n_enriched} sweeps with spike counts "
                            f"from raw traces")

                # Log summary of what we found
                spiking = [sw for sw in ls_sweeps
                           if (sw.get("num_spikes") or 0) > 0]
                if spiking:
                    best = max(spiking, key=lambda s: s["num_spikes"])
                    logger.info(
                        f"  Best spiking sweep: #{best['sweep_number']} "
                        f"({best.get('stimulus_amplitude', '?')} pA, "
                        f"{best['num_spikes']} spikes, "
                        f"method={best.get('spike_detection_method', '?')})")
                else:
                    logger.warning(f"  No spiking sweeps found even after "
                                   f"trace scanning!")
            except Exception as e:
                logger.error(f"  Failed to enrich spike counts: {e}")

        split = make_train_heldout_split(inventory)
        is_valid, reason = validate_cell(inventory, split)

        cell_entry = {
            "specimen_id": specimen_id,
            "inventory": inventory,
            "split": split,
            "valid": is_valid,
            "validation_reason": reason,
        }

        if not is_valid:
            logger.warning(f"  SKIPPED: {reason}")
            skipped_cells.append({"id": specimen_id, "reason": reason})
            sweep_index[str(specimen_id)] = cell_entry
            continue

        # Optionally download NWB and do a quick extraction sanity check
        if download_nwb:
            try:
                train_sweeps = [
                    sw["sweep_number"]
                    for sw in split["training"]["long_square"]
                ]
                # Download NWB (cached by Allen SDK) and extract first training sweep
                data_set = ctc.get_ephys_data(specimen_id)
                test_sw = extract_sweep(data_set, train_sweeps[0])
                cell_entry["nwb_verified"] = True
                cell_entry["sampling_rate"] = float(test_sw["sampling_rate"])
                cell_entry["first_sweep_duration_s"] = float(
                    len(test_sw["response"]) / test_sw["sampling_rate"]
                )
                logger.info(
                    f"  OK: {split['summary']['n_train_long_square']} train sweeps, "
                    f"sr={test_sw['sampling_rate']:.0f} Hz, "
                    f"duration={cell_entry['first_sweep_duration_s']:.2f}s"
                )
            except Exception as e:
                logger.error(f"  NWB verification failed: {e}")
                cell_entry["nwb_verified"] = False

        valid_cells.append(specimen_id)
        sweep_index[str(specimen_id)] = cell_entry

    # --- Step 5: Save sweep index ---
    # Convert numpy types for JSON serialization
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(cache_dir / "sweep_index.json", "w") as f:
        json.dump(sweep_index, f, indent=2, default=convert)

    # --- Summary ---
    print(f"\n{'='*70}")
    print("PIPELINE SUMMARY")
    print(f"{'='*70}")
    print(f"Total PV+ cells in database:  {len(pv_df)}")
    print(f"Cells processed:              {len(pv_ids)}")
    print(f"Valid cells (train + heldout): {len(valid_cells)}")
    print(f"Skipped cells:                {len(skipped_cells)}")
    print(f"\nData saved to: {cache_dir.resolve()}")
    print(f"  pv_cells_metadata.csv    — cell metadata")
    print(f"  pv_ephys_features.csv    — precomputed ephys features")
    print(f"  sweep_index.json         — per-cell sweep inventory & train/heldout splits")

    if valid_cells:
        # Print a summary table of valid cells
        print(f"\nValid cells ({len(valid_cells)}):")
        print(f"  {'Specimen ID':<15} {'Train LS':<10} {'Noise':<8} {'Ramp':<8} {'Short Sq':<10} {'LS Held-out':<12}")
        print(f"  {'-'*63}")
        for sid in valid_cells:
            s = sweep_index[str(sid)]["split"]["summary"]
            print(f"  {sid:<15} {s['n_train_long_square']:<10} "
                  f"{s['n_noise']:<8} {s['n_ramp']:<8} "
                  f"{s['n_short_square']:<10} {s['n_heldout_long_square']:<12}")

    return valid_cells, sweep_index


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download Allen Cell Types data for PV+ fast-spiking interneurons"
    )
    parser.add_argument(
        "--cache-dir", type=str, default="cell_types_data",
        help="Directory for Allen SDK cache and output files (default: cell_types_data)"
    )
    parser.add_argument(
        "--max-cells", type=int, default=None,
        help="Limit number of cells to process (for testing)"
    )
    parser.add_argument(
        "--list-only", action="store_true",
        help="Only list PV+ cells without downloading data"
    )
    parser.add_argument(
        "--no-nwb", action="store_true",
        help="Skip NWB download and verification (only get metadata and sweep lists)"
    )

    args = parser.parse_args()
    run_pipeline(
        cache_dir=args.cache_dir,
        max_cells=args.max_cells,
        list_only=args.list_only,
        download_nwb=not args.no_nwb,
    )