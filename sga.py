"""
SGA-Style Outer Loop (Weeks 7–8 / 11–12)
=========================================

Adapts the Scientific Generative Agent (Ma et al., NeurIPS 2024) bilevel
optimization framework from material science to neuroscience.

Core components:
    1. TopKHeap — maintains the k best model proposals ranked by loss
    2. ModelProposal — structured representation of a proposed HH model
    3. DiagnosticReport — translates inner-loop results into LLM feedback
    4. OuterLoop — orchestrates the propose → fit → diagnose → revise cycle
    5. Prompt templates — neuroscience-specific prompts for the LLM

Usage:
    from sga import OuterLoop

    loop = OuterLoop(
        specimen_id=509683388,
        data_dir="./cell_types_data",
        api_key="sk-ant-...",
        model="claude-sonnet-4-20250514",
    )
    best = loop.run(max_iterations=5)

Requires:
    pip install anthropic  # or openai
"""

import json
import time
import logging
import os
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional
import heapq

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logger = logging.getLogger(__name__)


# ===========================================================================
# 1. Model Proposal — what the LLM outputs
# ===========================================================================

@dataclass
class ModelProposal:
    """
    A proposed HH-type model structure. This is what the LLM generates
    and what gets passed to the Jaxley inner loop.
    """
    # Identity
    proposal_id: int = 0
    iteration: int = 0
    parent_id: Optional[int] = None

    # Model structure — list of channel names to insert
    channels: list = field(default_factory=lambda: ["Na", "K", "Leak"])

    # Parameter initial values and bounds
    # Format: {"param_name": {"init": float, "lower": float, "upper": float}}
    param_config: dict = field(default_factory=dict)

    # Cell geometry
    radius: float = 10.0
    length: float = 31.4
    capacitance: float = 1.0

    # LLM's rationale for this structure
    rationale: str = ""

    # Results (filled after inner loop)
    fitted_params: dict = field(default_factory=dict)
    loss: float = float("inf")
    diagnostics: dict = field(default_factory=dict)

    # For heap ordering (lower loss = better)
    def __lt__(self, other):
        return self.loss < other.loss

    def summary(self) -> str:
        return (
            f"Proposal #{self.proposal_id} (iter {self.iteration}): "
            f"channels=[{', '.join(self.channels)}] "
            f"loss={self.loss:.2f} "
            f"n_params={len(self.param_config)}"
        )


# ===========================================================================
# 2. Top-K Heap
# ===========================================================================

class TopKHeap:
    """Min-heap of the K best model proposals, ranked by loss."""

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
        if not self._heap:
            return None
        return min(self._heap, key=lambda p: p.loss)

    def worst_loss(self) -> float:
        if not self._heap:
            return float("inf")
        return max(p.loss for p in self._heap)

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
    """
    Structured feedback from the inner loop that the LLM uses to
    revise the model structure.
    """
    proposal: ModelProposal
    specimen_id: int

    # Inner loop results
    final_loss: float = float("inf")
    n_sim_spikes: int = 0
    n_target_spikes: int = 0
    pearson_r: float = 0.0
    model_spikes: bool = False

    # Diagnostic flags
    no_spikes: bool = False
    wrong_firing_rate: bool = False
    wrong_adaptation: bool = False
    excessive_sag: bool = False
    broad_spikes: bool = False
    parameters_at_bounds: list = field(default_factory=list)

    def generate_feedback(self) -> str:
        """Generate natural-language feedback for the LLM."""
        p = self.proposal
        lines = [
            f"## Inner Loop Results for Proposal #{p.proposal_id}",
            f"- Channels: {', '.join(p.channels)}",
            f"- Final MSE loss: {self.final_loss:.2f}",
            f"- Pearson correlation: {self.pearson_r:.3f}",
            f"- Simulated spikes: {self.n_sim_spikes}, "
            f"Target spikes: {self.n_target_spikes}",
        ]
    
        if p.fitted_params:
            lines.append("\n## Fitted Parameters (final values):")
            for k, v in p.fitted_params.items():
                lines.append(f"  {k}: {v:.6f}")

        # Show the bounds that were actually used for ALL parameters
        lines.append("\n## Bounds Used (from your param_config + defaults):")
        from general_fit import DEFAULT_PARAM_BOUNDS
        all_param_names = list(p.fitted_params.keys()) if p.fitted_params else []
        for name in all_param_names:
            if name in p.param_config:
                cfg = p.param_config[name]
                source = "your param_config"
            elif name in DEFAULT_PARAM_BOUNDS:
                cfg = DEFAULT_PARAM_BOUNDS[name]
                source = "default"
            else:
                continue
            fitted_val = p.fitted_params.get(name, "?")
            at_bound = ""
            lower = cfg.get('lower', '?')
            upper = cfg.get('upper', '?')
            if (isinstance(fitted_val, (int, float)) and isinstance(lower, (int, float))and isinstance(upper, (int, float))):
                margin = (upper - lower) * 0.02
                if abs(fitted_val - lower) < margin:
                    at_bound = " ⚠️ HIT LOWER BOUND"
                elif abs(fitted_val - upper) < margin:
                    at_bound = " ⚠️ HIT UPPER BOUND"
            lines.append(
                f"  {name}: fitted={fitted_val:.4f} in [{lower}, {upper}] "
                f"({source}){at_bound}"
            )
    
        lines.append("\n## Diagnostic Issues:")
    
        if self.no_spikes:
            lines.append(
                "- **MODEL DOES NOT SPIKE**: The simulation produced 0 spikes "
                "while the target has spikes. "
                "This suggests Na conductance is too low, the cell geometry gives "
                "insufficient current density, or a necessary depolarising current "
                "is missing. Check if Na_gNa or eNa are at their upper bounds — "
                "if so, you MUST widen those bounds in param_config."
            )
    
        if self.wrong_firing_rate:
            ratio = self.n_sim_spikes / max(self.n_target_spikes, 1)
            if ratio < 0.5:
                lines.append(
                    f"- **FIRING RATE TOO LOW**: {self.n_sim_spikes} vs "
                    f"{self.n_target_spikes} target. Consider: "
                    f"(1) widening Na_gNa upper bound in param_config, "
                    f"(2) widening eNa upper bound, "
                    f"(3) reducing K conductance bounds, "
                    f"(4) decreasing radius to increase current density."
                )
            else:
                lines.append(
                    f"- **FIRING RATE TOO HIGH**: {self.n_sim_spikes} vs "
                    f"{self.n_target_spikes} target. Consider: "
                    f"(1) adding adaptation channels (IM, IAHP), "
                    f"(2) increasing K conductance upper bound, "
                    f"(3) widening Kv3 bounds for faster repolarisation."
                )
    
        if self.broad_spikes:
            lines.append(
                "- **BROAD SPIKES**: Spike width is too large. Consider adding "
                "Kv3 for faster repolarisation."
            )
    
        if self.excessive_sag:
            lines.append(
                "- **EXCESSIVE SAG**: Model hyperpolarises too much. Consider "
                "reducing IH or adjusting leak reversal potential bounds."
            )
    
        if self.parameters_at_bounds:
            lines.append(f"\n- **PARAMETERS AT BOUNDS**: {self.parameters_at_bounds}")
            lines.append(
                "These parameters are stuck at their optimisation limits. "
                "The optimizer WANTS to push them further but CANNOT. "
                "You MUST widen the bounds for these parameters in your "
                "param_config. Simply changing channels will NOT fix this. "
                "For each parameter listed above, include it in your param_config "
                "with a wider range."
            )
    
        if not any([self.no_spikes, self.wrong_firing_rate,
                    self.broad_spikes, self.excessive_sag]):
            lines.append(
                "- No major structural issues detected. "
                "Consider fine-tuning parameter bounds or trying "
                "different channel combinations for marginal improvement."
            )
    
        return "\n".join(lines)


# ===========================================================================
# 4. Prompt Templates
# ===========================================================================

SYSTEM_PROMPT = """You are a computational neuroscience expert specialising in
Hodgkin-Huxley-type biophysical models of cortical neurons. You are working with
the Jaxley differentiable simulation framework to discover optimal HH model
structures for specific cortical neuron types from the Allen Cell Types Database.
 
Your task is to propose and iteratively refine the ion channel composition of
single-compartment neuron models. You have access to these channels:
 
BUILT-IN (Jaxley):
- Na: Standard Hodgkin-Huxley sodium channel (fast transient)
- K: Standard HH delayed rectifier potassium channel
- Leak: Passive leak conductance
 
CUSTOM LIBRARY:
- Kv3: Fast delayed rectifier K+ (Kv3/Shaw). Enables high-frequency firing in PV+ FS cells.
       Key param: Kv3_gKv3, typical range 1e-4 to 0.1 S/cm²
- IM: Muscarinic M-type K+ (KCNQ/Kv7). Produces spike-frequency adaptation.
      Key param: IM_gM, typical range 1e-6 to 1e-3 S/cm²
- IAHP: Ca²⁺-dependent K+ (medium AHP). Mediates afterhyperpolarisation.
        Key param: IAHP_gAHP, typical range 1e-6 to 1e-3 S/cm²
- IT: T-type Ca²⁺ (low-voltage-activated). Rebound bursting in SST+ cells.
      Key param: IT_gT, typical range 1e-5 to 1e-2 S/cm²
- ICaL: L-type Ca²⁺ (high-voltage-activated). Sustained calcium entry.
        Key param: ICaL_gCaL, typical range 1e-5 to 1e-2 S/cm²
- IH: HCN (I_h). Sag and rebound on hyperpolarisation.
      Key param: IH_gH, typical range 1e-6 to 1e-3 S/cm²
 
IMPORTANT CONSTRAINTS:
- Na, K, and Leak are always required and will be auto-inserted.
- Radius should be 5-15 µm (soma-like). Too large dilutes injected current.
- Capacitance should be 0.5-2.0 µF/cm². Too large slows dynamics.
- The optimizer uses sigmoid-bounded gradient descent. Don't set extreme bounds.
 
## DEFAULT PARAMETER BOUNDS
 
If you do NOT specify a parameter in param_config, these defaults are used:
 
  Na_gNa:      init=0.5,  lower=0.05,  upper=15.0
  K_gK:        init=0.2,  lower=0.01,  upper=2.0
  Leak_gLeak:  init=0.001, lower=1e-5, upper=0.01
  Leak_eLeak:  init=-65,  lower=-75,   upper=-50
  Kv3_gKv3:    init=0.01, lower=1e-4,  upper=0.1
  IM_gM:       init=1e-4, lower=1e-6,  upper=1e-3
  IAHP_gAHP:   init=1e-4, lower=1e-6,  upper=1e-3
  IT_gT:       init=1e-4, lower=1e-5,  upper=1e-2
  ICaL_gCaL:   init=1e-4, lower=1e-5,  upper=1e-2
  IH_gH:       init=1e-5, lower=1e-6,  upper=1e-3
  eNa:         init=50,   lower=40,    upper=70
  eK:          init=-77,  lower=-90,   upper=-70
  capacitance: init=1.0,  lower=0.5,   upper=2.0
  radius:      init=10.0, lower=5.0,   upper=15.0
 
## USING param_config TO OVERRIDE BOUNDS
 
**CRITICAL: When the diagnostic feedback reports a parameter "at bounds", you
MUST widen that parameter's bounds in your next param_config.** A parameter
stuck at its bound means the optimizer wants to push it further but cannot.
Simply changing channels will not fix this — you must give the optimizer room
to move.
 
Your param_config entries override the defaults above. You only need to include
parameters you want to change — anything omitted uses the default.
 
Each entry must have all three fields: "init", "lower", "upper".
 
## EXAMPLE
 
Suppose the feedback says:
  "Na_gNa at upper bound (5.0)" and "eNa at upper bound (60.0)"
 
This means the model needs more sodium current. A good response would widen
BOTH the conductance AND reversal potential bounds:
 
```json
{
    "channels": ["Na", "K", "Leak", "Kv3"],
    "param_config": {
        "Na_gNa":  {"init": 3.0, "lower": 0.1, "upper": 15.0},
        "K_gK":    {"init": 0.3, "lower": 0.01, "upper": 3.0},
        "Kv3_gKv3": {"init": 0.02, "lower": 1e-4, "upper": 0.5},
        "eNa":     {"init": 55.0, "lower": 45.0, "upper": 75.0},
        "eK":      {"init": -80.0, "lower": -100.0, "upper": -65.0}
    },
    "radius": 10.0,
    "capacitance": 1.0,
    "rationale": "Na_gNa was stuck at its 5.0 upper bound — widened to 15.0 to allow the optimizer to find the needed conductance. Also widened eNa upper to 75 mV since it was constrained at 60 mV. Kept Kv3 for fast repolarisation but widened its upper bound for flexibility."
}
```
 
## RESPONSE FORMAT
 
Always respond with a JSON object containing:
{
    "channels": ["Na", "K", "Leak", ...additional channels...],
    "param_config": {
        "param_name": {"init": float, "lower": float, "upper": float},
        ...
    },
    "radius": float,
    "capacitance": float,
    "rationale": "Your reasoning, especially explaining any bound changes..."
}
 
Include param_config entries for ALL parameters you have an opinion about,
especially any that were flagged as at-bounds in the feedback. Omitting a
parameter means accepting the default bounds listed above.
"""


def make_initial_prompt(neuron_metadata: dict, ephys_features: dict) -> str:
    """Build the Stage 1 prompt: propose initial model structure."""
    return f"""## Task: Propose an initial HH model structure

## Neuron Metadata:
{json.dumps(neuron_metadata, indent=2)}

## Electrophysiology Features:
{json.dumps(ephys_features, indent=2)}

Based on this neuron's properties, propose an initial set of ion channels
and parameter configurations. Consider:
1. Which channels are biologically expected for this neuron type?
2. What conductance densities are appropriate?
3. What cell geometry (radius) matches the expected soma size?

Respond with a JSON object as specified in your instructions."""


def make_revision_prompt(diagnostic, heap_summary: str) -> str:
    """Build the Stage 2 outer-loop prompt: revise model based on feedback."""
    feedback = diagnostic.generate_feedback()
 
    return f"""## Task: Revise the model structure based on fitting results
 
{feedback}
 
## Current Best Proposals:
{heap_summary}
 
Based on the diagnostic feedback above, propose a REVISED model structure.
You may:
- Add or remove channels to address identified issues
- **Widen parameter bounds via param_config for any parameter flagged as "at bounds"** — this is the HIGHEST PRIORITY action when parameters are constrained
- Adjust initial values to start closer to the expected optimum
- Modify cell geometry (radius, capacitance)
 
### Decision Priority:
1. FIRST: Fix any "parameters at bounds" by widening those bounds in param_config
2. THEN: Address structural issues (missing channels, wrong channel types)
3. LAST: Fine-tune initial values and geometry
 
Focus on the most critical issue first. Explain your reasoning, especially
for any bound changes.
Respond with a JSON object as specified in your instructions."""


# ===========================================================================
# 5. Outer Loop
# ===========================================================================

class OuterLoop:
    """
    The SGA-style outer loop that orchestrates:
    1. LLM proposes model structure (or revises based on feedback)
    2. Inner loop (Jaxley gradient descent) optimises parameters
    3. Diagnostics generated from fitting results
    4. Results pushed to top-K heap
    5. LLM receives feedback and proposes revision
    """

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

        # Resolve API key: explicit arg > env var (loaded from .env by dotenv)
        if api_key:
            self.api_key = api_key
        elif provider == "anthropic":
            self.api_key = os.environ.get("ANTHROPIC_API_KEY")
        else:
            self.api_key = os.environ.get("OPENAI_API_KEY")

        if not self.api_key:
            raise ValueError(
                "No API key found. Set ANTHROPIC_API_KEY in .env file, "
                "environment, or pass api_key= directly."
            )

    def _call_llm(self, system: str, user: str) -> str:
        """Call the LLM API. Supports Anthropic and OpenAI."""
        if self.provider == "anthropic":
            import anthropic
            client = anthropic.Anthropic(api_key=self.api_key)
            response = client.messages.create(
                model=self.model,
                max_tokens=2000,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            return response.content[0].text

        elif self.provider == "openai":
            import openai
            client = openai.OpenAI(api_key=self.api_key)
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                max_tokens=2000,
            )
            return response.choices[0].message.content

        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    def _parse_proposal(self, llm_response: str, iteration: int,
                        parent_id: Optional[int] = None) -> ModelProposal:
        """Parse LLM JSON response into a ModelProposal."""
        text = llm_response.strip()
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()

        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {e}")
            logger.error(f"Response was: {llm_response[:500]}")
            data = {"channels": ["Na", "K", "Leak"],
                    "rationale": f"JSON parse failed: {e}"}

        return ModelProposal(
            proposal_id=self.heap.next_id(),
            iteration=iteration,
            parent_id=parent_id,
            channels=data.get("channels", ["Na", "K", "Leak"]),
            param_config=data.get("param_config", {}),
            radius=data.get("radius", 10.0),
            length=data.get("length", 31.4),
            capacitance=data.get("capacitance", 1.0),
            rationale=data.get("rationale", ""),
        )

    def _run_inner_loop(self, proposal: ModelProposal) -> DiagnosticReport:
        """
        Run the Jaxley inner loop for a proposal.
        Calls general_fit.fit_proposal() which dynamically builds a
        Jaxley neuron, runs gradient descent, returns a DiagnosticReport.
        """
        logger.info(f"  Inner loop for proposal #{proposal.proposal_id}: "
                    f"{proposal.channels}")
        logger.info(f"  Rationale: {proposal.rationale[:200]}")

        from general_fit import fit_proposal

        return fit_proposal(
            proposal=proposal,
            specimen_id=self.specimen_id,
            data_dir=str(self.data_dir),
            epochs=self.inner_epochs,
            lr=self.inner_lr,
        )

    def run(self, max_iterations: int = 5,
            neuron_metadata: dict = None,
            ephys_features: dict = None) -> ModelProposal:
        """
        Run the full outer loop for up to max_iterations.

        Each iteration:
          1. LLM proposes (iter 0) or revises (iter >0) model structure
          2. Inner loop fits the proposed model via gradient descent
          3. Diagnostics are computed and fed back to the LLM
          4. Result is pushed to the top-K heap

        Returns the best proposal from the heap.
        """
        if neuron_metadata is None:
            neuron_metadata = {
                "cell_type": "PV+ fast-spiking interneuron",
                "transgenic_line": "Pvalb-IRES-Cre",
                "dendrite_type": "aspiny",
                "cortical_layer": "4",
                "brain_region": "VISp",
            }
        if ephys_features is None:
            ephys_features = {
                "note": "Load from pv_ephys_features.csv for this specimen",
            }

        logger.info(f"Starting outer loop: max {max_iterations} iterations, "
                    f"specimen {self.specimen_id}")

        last_diagnostic = None

        for iteration in range(max_iterations):
            logger.info(f"\n{'='*60}")
            logger.info(f"OUTER LOOP ITERATION {iteration + 1}/{max_iterations}")
            logger.info(f"{'='*60}")

            # ---- Step 1: LLM proposes (or revises) model ----
            if iteration == 0:
                prompt = make_initial_prompt(neuron_metadata, ephys_features)
                parent_id = None
            else:
                # Use the diagnostic from the PREVIOUS iteration
                prompt = make_revision_prompt(
                    last_diagnostic, self.heap.summary())
                parent_id = last_diagnostic.proposal.proposal_id

            logger.info("  Calling LLM for proposal...")
            try:
                response = self._call_llm(SYSTEM_PROMPT, prompt)
                proposal = self._parse_proposal(response, iteration, parent_id)
                logger.info(f"  LLM proposed: {proposal.channels}")
                logger.info(f"  Rationale: {proposal.rationale[:300]}")
            except Exception as e:
                logger.error(f"  LLM call failed: {e}")
                # On LLM failure, retry with a safe default
                proposal = ModelProposal(
                    proposal_id=self.heap.next_id(),
                    iteration=iteration,
                    channels=["Na", "K", "Leak", "Kv3"],
                    rationale=f"Fallback after LLM error: {e}",
                )

            # ---- Step 2: Inner loop — fit the proposal ----
            try:
                diagnostic = self._run_inner_loop(proposal)
            except Exception as e:
                logger.error(f"  Inner loop crashed: {e}")
                diagnostic = DiagnosticReport(
                    proposal=proposal,
                    specimen_id=self.specimen_id,
                    final_loss=float("inf"),
                    no_spikes=True,
                )

            proposal.loss = diagnostic.final_loss
            proposal.diagnostics = asdict(diagnostic)
            last_diagnostic = diagnostic

            # ---- Step 3: Push to heap ----
            self.heap.push(proposal)

            # ---- Step 4: Log history ----
            self.history.append({
                "iteration": iteration,
                "proposal": asdict(proposal),
                "diagnostic_feedback": diagnostic.generate_feedback(),
            })

            logger.info(f"\n  Loss: {diagnostic.final_loss:.2f}, "
                        f"Spikes: {diagnostic.n_sim_spikes}/{diagnostic.n_target_spikes}, "
                        f"r={diagnostic.pearson_r:.3f}")
            logger.info(f"\n{self.heap.summary()}")

            # ---- Step 5: Early stopping ----
            best = self.heap.best()
            if best and best.loss < float("inf"):
                best_diag = best.diagnostics
                has_spikes = best_diag.get("model_spikes", False)
                n_sim = best_diag.get("n_sim_spikes", 0)
                n_tgt = best_diag.get("n_target_spikes", 1)
                if has_spikes and n_tgt > 0:
                    rate_error = abs(n_sim - n_tgt) / n_tgt
                    if rate_error < 0.2 and best_diag.get("pearson_r", 0) > 0.8:
                        logger.info(
                            f"  ✓ Good fit found (rate_error={rate_error:.2f}, "
                            f"r={best_diag.get('pearson_r', 0):.3f}) — stopping early.")
                        break

        # ---- Save history ----
        history_path = self.data_dir / "sga_history.json"
        with open(history_path, "w") as f:
            json.dump(self.history, f, indent=2, default=str)
        logger.info(f"History saved to {history_path}")

        return self.heap.best()


# ===========================================================================
# CLI for testing
# ===========================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("SGA Scaffold — Components Test")
    print("=" * 50)

    # Test TopKHeap
    heap = TopKHeap(k=3)
    for i in range(5):
        p = ModelProposal(
            proposal_id=i,
            channels=["Na", "K", "Leak"] + (["Kv3"] if i % 2 == 0 else []),
            loss=100 - i * 10,
            rationale=f"Test proposal {i}",
        )
        heap.push(p)
    print(heap.summary())

    # Test DiagnosticReport
    best = heap.best()
    diag = DiagnosticReport(
        proposal=best,
        specimen_id=509683388,
        final_loss=best.loss,
        n_sim_spikes=4,
        n_target_spikes=56,
        no_spikes=False,
        wrong_firing_rate=True,
    )
    print("\n" + diag.generate_feedback())

    print("\n✓ Component test passed")