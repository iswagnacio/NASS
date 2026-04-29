"""
Run held-out validation with the fitted params from the 300-epoch
two-compartment training run.

Usage:
    cd ~/NASS
    CUDA_VISIBLE_DEVICES=1 python run_validation_2comp.py
"""

from jax import config
config.update("jax_enable_x64", True)

from validation import run_held_out_validation

# Fitted params from the 300-epoch run (2026-04-28, best loss=3538.07)
proposal = {
    "channels": ["Na", "K", "Leak", "Kv3"],
    "radius": 10.0,
    "length": 31.4,
    "capacitance": 1.0,
    "fitted_params": {
        "Na_gNa": 0.0005544855469189955,
        "K_gK": 0.00631958728257926,
        "Leak_gLeak": 0.00017663875021475277,
        "Leak_eLeak": -71.8881736497965,
        "Kv3_gKv3": 0.005856581151072926,
        "eNa": 49.96400653806753,
        "eK": -89.6338249227818,
        "capacitance": 1.0633173037856782,
        "radius": 10.830518990454316,
    },
    "loss": 3538.0689,
    "proposal_id": 0,
}

SPECIMEN_ID = 469801138
DATA_DIR = "cell_types_data"

report = run_held_out_validation(
    specimen_id=SPECIMEN_ID,
    data_dir=DATA_DIR,
    proposal=proposal,
    dt=0.025,
    max_sweeps_per_category=3,
    save_plots=True,
)

if report.results:
    overall = report.aggregates.get("overall", {})
    gamma = overall.get("spike_coincidence_mean", 0)
    print(f"\nValidation complete. Overall Gamma = {gamma:.3f}")
else:
    print("\nValidation produced no results.")