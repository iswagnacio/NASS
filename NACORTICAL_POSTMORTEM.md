# NaCortical Implementation Postmortem

**Date:** March 27, 2026
**Status:** FAILED - Reverted to standard HH Na

## Summary

Attempted to implement a cortical-specific sodium channel (NaCortical) with 2-3× faster h-gate recovery to enable high-frequency firing in PV+ fast-spiking interneurons. The implementation failed catastrophically and has been reverted.

## Results Comparison

| Implementation | Spikes (out of 56) | NaN Gradients | Status |
|---|---|---|---|
| **Pre-refactor (standard HH Na)** | 44 | Occasional | ✓ Best |
| **NaCortical v1 (wrong acceleration)** | 2 | 400/400 epochs | ✗ Failed |
| **NaCortical v2 (fixed acceleration)** | 0 | 200/200 epochs | ✗✗ Worse |
| **NaCortical v3 (reduced accel, higher floor)** | 0 | 30/200 epochs | ✗✗ Still broken |

## What Was Tried

### Attempt 1: Initial Implementation
- tau_h accelerated by 1.5-3×
- **Bug:** Acceleration applied during spikes instead of at rest
- Result: 2 spikes, massive NaN gradients

### Attempt 2: Fixed Acceleration Direction
- Changed to: `accel = 1.0 + 1.5 * (1 - sigmoid(v, -40, -10))`
- Correctly accelerates at rest, not during spikes
- Result: 0 spikes, still 200 NaN gradients

### Attempt 3: Conservative Parameters
- Reduced acceleration: 2× instead of 2.5×
- Raised tau_h floor: 0.3 ms instead of 0.1 ms
- Result: 0 spikes, 30 NaN gradients (improved but still broken)

## Root Cause Analysis

### Numerical Instability
The channel produces NaN gradients regularly, suggesting:
1. **ODE stiffness**: Even with tau_h ≥ 0.3 ms, the system is too stiff for Jaxley's integrator
2. **m-gate/h-gate mismatch**: Traub & Miles cortical m-gate doesn't balance with accelerated h-gate
3. **Backprop amplification**: Small numerical errors compound through 45,999 timesteps of BPTT

### Complete Spiking Failure
With parameters at their limits (eNa=115, eK=-120, Na_gNa=0.39, radius=20), the model produces:
- High correlation (r=0.657) = learns subthreshold dynamics
- Zero spikes = never crosses threshold
- Suggests the channel enters an inactivated state and can't recover

## Why Standard HH Na Works Better

The pre-refactor with standard HH Na achieved 44/56 spikes. This suggests:

1. **The report's diagnosis may be wrong**: The bottleneck might not be Na channel kinetics
2. **Optimizer exploitation**: With eNa constrained to 80 mV, the optimizer found a working solution
3. **Parameter space**: Standard HH has a larger viable parameter space for single-compartment models

## Lessons Learned

1. **Channel kinetics are delicate**: Modifying one gate (h) without re-balancing the other (m) breaks the channel
2. **Single-compartment constraints**: What works in multi-compartment models doesn't translate
3. **Test before integrating**: Should have validated NaCortical in isolation before putting it in the pipeline
4. **Trust the baseline**: 44/56 spikes is actually pretty good for single-compartment HH

## What's Actually Wrong?

Looking at the pre-refactor results:
- 44 spikes with eNa=80 mV cap ✓
- 22 spikes with eNa=115 mV cap ✗

The problem isn't that standard HH can't produce spikes — it produced 44! The problem is:

1. **Radius inflation**: Optimizer pushes radius to upper bound (20 µm) to reduce excitability
2. **Loss function conflicts**: Spike count loss vs. spike timing loss create opposing gradients
3. **Geometry constraints**: Single compartment can't separate fast AIS spiking from slow somatic integration

## Recommended Next Steps

### Option 1: Tighter Radius Constraints (Recommended)
- Cap radius at 8-12 µm (not 20 µm)
- Force high current density
- May recover 44-spike performance

### Option 2: Option B from Report - Two-Compartment Model
- AIS + soma with separate Na densities
- Matches real neuron architecture
- More complex but more realistic

### Option 3: Phased Loss Training
- Train spike count only for first 200 epochs
- Add timing loss after count matches
- Avoid conflicting gradients in partial-spiking regime

### Option 4: Different Channel (not kinetics)
- Maybe the issue isn't Na recovery speed
- Try adding NaP (persistent Na) for sustained drive
- Or Kv1 for better repolarization balance

## Files Reverted

- `general_fit.py`: Na → standard Jaxley Na (was NaCortical)
- `sim_fit.py`: Na → standard Jaxley Na (was NaCortical)
- `sga.py`: Removed NaCortical documentation
- `FALLBACK_PARAM_BOUNDS`: Na_gNa → [0.01, 0.20] (was [0.05, 0.50])

## Archival

The NaCortical implementation remains in `channels.py` for future investigation:
- Can be accessed as "NaCortical" in channel lists
- Documentation preserved
- May be useful for multi-compartment models or different integration schemes

## Conclusion

**The NaCortical approach failed.** Standard HH Na with better parameter constraints is the path forward. The report correctly identified that standard HH limits firing rate, but our "solution" made things worse by introducing numerical instability.

The real bottleneck is likely **geometry** (radius inflation) and **loss function design**, not channel kinetics.

---

**Next action**: Run with standard HH Na and tighter radius bounds to validate pre-refactor 44-spike performance.
