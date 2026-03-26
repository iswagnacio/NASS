"""
SGA-Style Outer Loop
====================

Adapts the Scientific Generative Agent (Ma et al., NeurIPS 2024) bilevel
optimization framework for neuroscience.

Architecture (after auto_bounds refactor):
    - Trace features extracted ONCE at start, included in every LLM prompt
    - LLM makes ALL biophysical decisions (bounds, inits, channels)
    - Inner loop only applies gradient safety clamping (NaN prevention)
    - FALLBACK_PARAM_BOUNDS used only when LLM omits a parameter
"""

import json
import time
import logging
import os
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional
import heapq
import numpy as np

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logger = logging.getLogger(__name__)


# ===========================================================================
# 1. Model Proposal
# ===========================================================================

@dataclass
class ModelProposal:
    proposal_id: int = 0
    iteration: int = 0
    parent_id: Optional[int] = None
    channels: list = field(default_factory=lambda: ["Na", "K", "Leak"])
    param_config: dict = field(default_factory=dict)
    radius: float = 10.0
    length: float = 31.4
    capacitance: float = 1.0
    rationale: str = ""
    fitted_params: dict = field(default_factory=dict)
    loss: float = float("inf")
    diagnostics: dict = field(default_factory=dict)

    def __lt__(self, other):
        return self.loss < other.loss

    def summary(self) -> str:
        return (f"Proposal #{self.proposal_id} (iter {self.iteration}): "
                f"channels=[{', '.join(self.channels)}] "
                f"loss={self.loss:.2f} n_params={len(self.param_config)}")


# ===========================================================================
# 2. Top-K Heap
# ===========================================================================

class TopKHeap:
    def __init__(self, k: int = 5):
        self.k = k
        self._heap: list[ModelProposal] = []
        self._counter = 0

    def push(self, proposal: ModelProposal):
        if len(self._heap) < self.k:
            heapq.heappush(self._heap, proposal)
        elif proposal.loss < self.worst_loss():
            heapq.heapreplace(self._heap, proposal)

    def best(self) -> Optional[ModelProposal]:
        return min(self._heap, key=lambda p: p.loss) if self._heap else None

    def worst_loss(self) -> float:
        return max(p.loss for p in self._heap) if self._heap else float("inf")

    def top_k(self) -> list[ModelProposal]:
        return sorted(self._heap, key=lambda p: p.loss)

    def next_id(self) -> int:
        self._counter += 1
        return self._counter

    def __len__(self):
        return len(self._heap)

    def summary(self) -> str:
        lines = [f"TopKHeap ({len(self)}/{self.k} proposals):"]
        for p in self.top_k():
            lines.append(f"  {p.summary()}")
        return "\n".join(lines)


# ===========================================================================
# 3. Diagnostic Report
# ===========================================================================

@dataclass
class DiagnosticReport:
    proposal: ModelProposal
    specimen_id: int
    final_loss: float = float("inf")
    n_sim_spikes: int = 0
    n_target_spikes: int = 0
    pearson_r: float = 0.0
    model_spikes: bool = False
    no_spikes: bool = False
    wrong_firing_rate: bool = False
    wrong_adaptation: bool = False
    excessive_sag: bool = False
    broad_spikes: bool = False
    parameters_at_bounds: list = field(default_factory=list)

    def generate_feedback(self) -> str:
        p = self.proposal
        lines = [
            f"## Inner Loop Results for Proposal #{p.proposal_id}",
            f"- Channels: {', '.join(p.channels)}",
            f"- Final loss: {self.final_loss:.2f}",
            f"- Pearson correlation: {self.pearson_r:.3f}",
            f"- Simulated spikes: {self.n_sim_spikes}, Target spikes: {self.n_target_spikes}",
        ]

        if p.fitted_params:
            lines.append("\n## Fitted Parameters (final values):")
            for k, v in p.fitted_params.items():
                lines.append(f"  {k}: {v:.6f}")

        lines.append("\n## Bounds Used (from your param_config + fallback defaults):")
        try:
            from general_fit import FALLBACK_PARAM_BOUNDS as _DEFAULTS
        except ImportError:
            _DEFAULTS = {}

        for name in (list(p.fitted_params.keys()) if p.fitted_params else []):
            if name in p.param_config:
                cfg, source = p.param_config[name], "your param_config"
            elif name in _DEFAULTS:
                cfg, source = _DEFAULTS[name], "fallback default"
            else:
                continue
            fitted_val = p.fitted_params.get(name, "?")
            at_bound = ""
            lower, upper = cfg.get('lower', '?'), cfg.get('upper', '?')
            if all(isinstance(x, (int, float)) for x in [fitted_val, lower, upper]):
                margin = (upper - lower) * 0.02
                if abs(fitted_val - lower) < margin:
                    at_bound = " ⚠️ HIT LOWER BOUND"
                elif abs(fitted_val - upper) < margin:
                    at_bound = " ⚠️ HIT UPPER BOUND"
            lines.append(f"  {name}: fitted={fitted_val:.4f} in [{lower}, {upper}] ({source}){at_bound}")

        lines.append("\n## Diagnostic Issues:")

        if self.no_spikes:
            lines.append(
                "- **MODEL DOES NOT SPIKE**: 0 spikes while target has spikes. "
                "Check if Na_gNa init is >= 0.10 (required for single-compartment spiking). "
                "Check if eNa is high enough (must exceed spike peak by 30+ mV). "
                "Check if radius is too large (dilutes current density).")

        if self.wrong_firing_rate:
            ratio = self.n_sim_spikes / max(self.n_target_spikes, 1)
            if ratio < 0.5:
                lines.append(
                    f"- **FIRING RATE TOO LOW**: {self.n_sim_spikes} vs {self.n_target_spikes}. "
                    f"Consider: wider Na_gNa/eNa bounds, smaller radius, lower capacitance.")
            else:
                lines.append(
                    f"- **FIRING RATE TOO HIGH**: {self.n_sim_spikes} vs {self.n_target_spikes}. "
                    f"Consider: adding IM/IAHP, wider K/Kv3 bounds, larger radius.")

        if self.broad_spikes:
            lines.append("- **BROAD SPIKES**: Consider adding Kv3 for faster repolarisation.")

        if self.excessive_sag:
            lines.append("- **EXCESSIVE SAG**: Reduce IH or adjust Leak_eLeak.")

        if self.parameters_at_bounds:
            lines.append(f"\n- **PARAMETERS AT BOUNDS**: {self.parameters_at_bounds}")
            lines.append(
                "These parameters are stuck at their optimisation limits. "
                "You MUST widen the bounds for these in your param_config.")

        if not any([self.no_spikes, self.wrong_firing_rate, self.broad_spikes, self.excessive_sag]):
            lines.append("- No major structural issues detected.")

        return "\n".join(lines)


# ===========================================================================
# 4. Prompt Templates
# ===========================================================================

SYSTEM_PROMPT = """You are a computational neuroscience expert specialising in
Hodgkin-Huxley biophysical models. You work with the Jaxley differentiable
simulation framework to discover optimal single-compartment HH models for
cortical neurons from the Allen Cell Types Database.

## Your Responsibility
You make ALL biophysical decisions: channel selection, parameter bounds, initial
values, and cell geometry. The system only overrides your choices for gradient
safety (preventing NaN in the optimizer). You receive measured trace features
from the target recording — use them to make informed choices from iteration 1.

## Available Channels

BUILT-IN: Na (fast sodium), K (delayed rectifier), Leak (passive)
CUSTOM: Kv3 (fast K+, FS cells), IM (M-type, adaptation), IAHP (AHP),
        IT (T-type Ca2+, bursting), ICaL (L-type Ca2+), IH (HCN, sag)

Na, K, and Leak are always auto-inserted.

## Critical Single-Compartment Modeling Rules

1. Na_gNa init MUST be >= 0.10 S/cm². Below this, Kv3 suppresses all spiking
   and every gradient is NaN. This is a Jaxley single-compartment requirement
   (real neurons have concentrated Na at the AIS; one compartment needs more).

2. eNa must be 30-90 mV ABOVE the spike peak. The spike peak is set by
   Na inactivation + K competition, NOT by proximity to eNa. Typical: 50-115 mV.

3. eK must be below the AHP trough. Typical: -120 to -70 mV.

4. Radius controls current density: smaller radius → higher current density →
   easier to spike. For fast-spiking cells, prefer 5-12 µm. The optimizer WILL
   inflate radius to suppress spiking if allowed — cap the upper bound.

5. Capacitance > 2.0 µF/cm² slows dynamics and reduces firing rate.

6. All conductances are in S/cm² (NOT mS/cm²). Classic HH Na = 0.12 S/cm².

## Gradient Safety Limits (the ONLY programmatic override)

These are hard limits enforced by the system to prevent NaN. Your param_config
should stay within these, but if you exceed them, they'll be silently clamped:

  Na_gNa upper:    ~0.47 S/cm² (scales with spike count)
  K_gK upper:      ~0.09 S/cm²
  Kv3_gKv3 upper:  ~0.09 S/cm²
  eNa:             [30, 115] mV
  eK:              [-120, -55] mV
  capacitance:     [0.2, 3.0] µF/cm²
  radius:          [1.5, 30] µm

## Response Format

```json
{
    "channels": ["Na", "K", "Leak", ...],
    "param_config": {
        "param_name": {"init": float, "lower": float, "upper": float},
        ...
    },
    "radius": float,
    "capacitance": float,
    "rationale": "Your reasoning..."
}
```

Include param_config for ALL parameters you have an opinion about.
Parameters you omit use conservative fallback defaults (which may be suboptimal).
The more parameters you specify, the better the fit will be.
"""


def make_initial_prompt(neuron_metadata: dict, ephys_features: dict,
                        trace_features_text: str = "") -> str:
    """Build the initial prompt with trace features."""
    return f"""## Task: Propose an initial HH model structure

## Neuron Metadata:
{json.dumps(neuron_metadata, indent=2)}

## Electrophysiology Features (Allen precomputed):
{json.dumps(ephys_features, indent=2)}

{trace_features_text}

Based on this neuron's properties AND the measured trace features above,
propose an initial set of ion channels and COMPLETE parameter configurations.

You MUST include param_config entries for at least: Na_gNa, K_gK, Leak_gLeak,
Leak_eLeak, eNa, eK, capacitance, and radius. Use the trace features to set
appropriate bounds (e.g., eNa must exceed spike peak, eK must be below AHP trough).

Respond with a JSON object as specified in your instructions."""


def make_revision_prompt(diagnostic, heap_summary: str,
                         trace_features_text: str = "") -> str:
    """Build the revision prompt with trace features."""
    feedback = diagnostic.generate_feedback()

    return f"""## Task: Revise the model structure based on fitting results

{feedback}

{trace_features_text}

## Current Best Proposals:
{heap_summary}

Based on the diagnostic feedback AND the trace features above, propose a
REVISED model structure. Priority:
1. Fix any "parameters at bounds" by widening those bounds in param_config
2. Address structural issues (missing channels, wrong types)
3. Fine-tune initial values and geometry

Respond with a JSON object as specified in your instructions."""


# ===========================================================================
# 5. Outer Loop
# ===========================================================================

class OuterLoop:
    def __init__(self, specimen_id: int, data_dir: str,
                 api_key: str = None, model: str = "claude-sonnet-4-20250514",
                 top_k: int = 5, provider: str = "anthropic",
                 inner_epochs: int = 300, inner_lr: float = 0.02):
        self.specimen_id = specimen_id
        self.data_dir = Path(data_dir)
        self.model = model
        self.provider = provider
        self.heap = TopKHeap(k=top_k)
        self.history: list[dict] = []
        self.inner_epochs = inner_epochs
        self.inner_lr = inner_lr
        self.trace_features_text = ""  # Populated once at start of run()

        if api_key:
            self.api_key = api_key
        elif provider == "anthropic":
            self.api_key = os.environ.get("ANTHROPIC_API_KEY")
        else:
            self.api_key = os.environ.get("OPENAI_API_KEY")

        if not self.api_key:
            raise ValueError("No API key found. Set ANTHROPIC_API_KEY in .env or pass api_key=.")

    def _extract_trace_features(self) -> str:
        """
        Extract trace features from the training sweep ONCE.
        Returns formatted text for inclusion in LLM prompts.
        """
        from auto_bounds import extract_trace_features, format_features_for_prompt
        from general_fit import extract_baseline
        from sim_fit import load_training_sweep, prepare_stimulus, prepare_target
        from allensdk.core.cell_types_cache import CellTypesCache

        try:
            ctc = CellTypesCache(manifest_file=str(self.data_dir / "manifest.json"))
            with open(self.data_dir / "sweep_index.json") as f:
                sweep_index = json.load(f)

            sweep = load_training_sweep(ctc, self.specimen_id, sweep_index)
            dt = 0.025
            stimulus, t_max = prepare_stimulus(sweep, dt)
            target_v = prepare_target(sweep, dt)

            # Extract baseline from pre-stimulus period
            baseline_v = extract_baseline(stimulus, target_v, dt)
            logger.info(f"  Baseline: {len(baseline_v)} samples, Vrest={np.mean(baseline_v):.1f} mV")

            # Window to stimulus region for spike analysis
            from general_fit import window_to_main_stimulus
            import jax.numpy as jnp
            stimulus_w, target_w, _ = window_to_main_stimulus(
                stimulus, jnp.array(target_v), dt)

            features = extract_trace_features(
                np.array(target_w), dt, baseline_v=baseline_v)

            return format_features_for_prompt(features)

        except Exception as e:
            logger.warning(f"  Could not extract trace features: {e}")
            return "## Trace Features: unavailable (data loading failed)"

    def _call_llm(self, system: str, user: str) -> str:
        if self.provider == "anthropic":
            import anthropic
            client = anthropic.Anthropic(api_key=self.api_key)
            response = client.messages.create(
                model=self.model, max_tokens=2000, system=system,
                messages=[{"role": "user", "content": user}])
            return response.content[0].text
        elif self.provider == "openai":
            import openai
            client = openai.OpenAI(api_key=self.api_key)
            response = client.chat.completions.create(
                model=self.model, max_tokens=2000,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}])
            return response.choices[0].message.content
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    def _parse_proposal(self, llm_response: str, iteration: int,
                        parent_id: Optional[int] = None) -> ModelProposal:
        text = llm_response.strip()
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()

        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM JSON: {e}")
            data = {"channels": ["Na", "K", "Leak"], "rationale": f"JSON parse failed: {e}"}

        return ModelProposal(
            proposal_id=self.heap.next_id(), iteration=iteration, parent_id=parent_id,
            channels=data.get("channels", ["Na", "K", "Leak"]),
            param_config=data.get("param_config", {}),
            radius=data.get("radius", 10.0), length=data.get("length", 31.4),
            capacitance=data.get("capacitance", 1.0), rationale=data.get("rationale", ""))

    def _run_inner_loop(self, proposal: ModelProposal) -> DiagnosticReport:
        logger.info(f"  Inner loop for proposal #{proposal.proposal_id}: {proposal.channels}")
        logger.info(f"  Rationale: {proposal.rationale[:200]}")
        from general_fit import fit_proposal
        return fit_proposal(proposal=proposal, specimen_id=self.specimen_id,
                            data_dir=str(self.data_dir), epochs=self.inner_epochs,
                            lr=self.inner_lr)

    def run(self, max_iterations: int = 5,
            neuron_metadata: dict = None,
            ephys_features: dict = None) -> ModelProposal:
        if neuron_metadata is None:
            neuron_metadata = {
                "cell_type": "PV+ fast-spiking interneuron",
                "transgenic_line": "Pvalb-IRES-Cre",
                "dendrite_type": "aspiny",
                "cortical_layer": "4", "brain_region": "VISp",
            }
        if ephys_features is None:
            ephys_features = {"note": "Load from pv_ephys_features.csv for this specimen"}

        logger.info(f"Starting outer loop: max {max_iterations} iterations, specimen {self.specimen_id}")

        # ---- Extract trace features ONCE, include in every prompt ----
        logger.info("  Extracting trace features for LLM prompts...")
        self.trace_features_text = self._extract_trace_features()
        logger.info(f"  Trace features extracted successfully")

        last_diagnostic = None

        for iteration in range(max_iterations):
            logger.info(f"\n{'='*60}")
            logger.info(f"OUTER LOOP ITERATION {iteration + 1}/{max_iterations}")
            logger.info(f"{'='*60}")

            if iteration == 0:
                prompt = make_initial_prompt(neuron_metadata, ephys_features,
                                            self.trace_features_text)
                parent_id = None
            else:
                prompt = make_revision_prompt(last_diagnostic, self.heap.summary(),
                                             self.trace_features_text)
                parent_id = last_diagnostic.proposal.proposal_id

            logger.info("  Calling LLM for proposal...")
            try:
                response = self._call_llm(SYSTEM_PROMPT, prompt)
                proposal = self._parse_proposal(response, iteration, parent_id)
                logger.info(f"  LLM proposed: {proposal.channels}")
                logger.info(f"  Rationale: {proposal.rationale[:300]}")
            except Exception as e:
                logger.error(f"  LLM call failed: {e}")
                proposal = ModelProposal(
                    proposal_id=self.heap.next_id(), iteration=iteration,
                    channels=["Na", "K", "Leak", "Kv3"],
                    rationale=f"Fallback after LLM error: {e}")

            try:
                diagnostic = self._run_inner_loop(proposal)
            except Exception as e:
                logger.error(f"  Inner loop crashed: {e}")
                diagnostic = DiagnosticReport(
                    proposal=proposal, specimen_id=self.specimen_id,
                    final_loss=float("inf"), no_spikes=True)

            proposal.loss = diagnostic.final_loss
            proposal.diagnostics = asdict(diagnostic)
            last_diagnostic = diagnostic

            self.heap.push(proposal)
            self.history.append({
                "iteration": iteration,
                "proposal": asdict(proposal),
                "diagnostic_feedback": diagnostic.generate_feedback(),
            })

            logger.info(f"\n  Loss: {diagnostic.final_loss:.2f}, "
                        f"Spikes: {diagnostic.n_sim_spikes}/{diagnostic.n_target_spikes}, "
                        f"r={diagnostic.pearson_r:.3f}")
            logger.info(f"\n{self.heap.summary()}")

            # Early stopping
            best = self.heap.best()
            if best and best.loss < float("inf"):
                bd = best.diagnostics
                if bd.get("model_spikes") and bd.get("n_target_spikes", 1) > 0:
                    rate_err = abs(bd.get("n_sim_spikes", 0) - bd["n_target_spikes"]) / bd["n_target_spikes"]
                    if rate_err < 0.2 and bd.get("pearson_r", 0) > 0.8:
                        logger.info(f"  ✓ Good fit found — stopping early.")
                        break

        history_path = self.data_dir / "sga_history.json"
        with open(history_path, "w") as f:
            json.dump(self.history, f, indent=2, default=str)
        logger.info(f"History saved to {history_path}")
        return self.heap.best()


# ===========================================================================
# CLI
# ===========================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("SGA Scaffold — Components Test")
    heap = TopKHeap(k=3)
    for i in range(5):
        p = ModelProposal(proposal_id=i, channels=["Na", "K", "Leak"] + (["Kv3"] if i % 2 == 0 else []),
                          loss=100 - i * 10, rationale=f"Test {i}")
        heap.push(p)
    print(heap.summary())
    best = heap.best()
    diag = DiagnosticReport(proposal=best, specimen_id=509683388, final_loss=best.loss,
                            n_sim_spikes=4, n_target_spikes=56, wrong_firing_rate=True)
    print("\n" + diag.generate_feedback())
    print("\n✓ Component test passed")