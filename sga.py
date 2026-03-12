"""
SGA-Style Outer Loop Scaffolding (Weeks 7–8, Part 2)
=====================================================

Adapts the Scientific Generative Agent (Ma et al., NeurIPS 2024) bilevel
optimization framework from material science to neuroscience.

Core components:
    1. TopKHeap — maintains the k best model proposals ranked by loss
    2. ModelProposal — structured representation of a proposed HH model
    3. DiagnosticReport — translates inner-loop results into LLM feedback
    4. OuterLoop — orchestrates the propose → fit → diagnose → revise cycle
    5. Prompt templates — neuroscience-specific prompts for the LLM

The key insight from the proposal: "SGA gives us ~70% of the infrastructure.
We replace its material-science simulation backend with Jaxley's neuron ODE
solver, replace its constitutive-law templates with HH equation templates,
and point it at Allen patch-clamp data instead of material deformation data."

Usage:
    from sga_scaffold import OuterLoop, ModelProposal

    loop = OuterLoop(
        specimen_id=509683388,
        data_dir="./data",
        api_key="your-api-key",
        model="claude-sonnet-4-20250514",
    )
    best = loop.run(max_iterations=5)

Requires:
    pip install anthropic  # or openai
"""

import json
import time
import logging
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional
import heapq

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
    parent_id: Optional[int] = None  # which proposal this was derived from

    # Model structure — list of channel names to insert
    channels: list = field(default_factory=lambda: ["Na", "K", "Leak"])

    # Parameter initial values and bounds
    # Format: {"channel_param_name": {"init": float, "lower": float, "upper": float}}
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

    def to_jaxley_code(self) -> str:
        """Generate Jaxley-compatible Python code for this model."""
        lines = [
            "import jaxley as jx",
            "from jaxley.channels import Na, K, Leak",
            "from channels import Kv3, IM, IAHP, IT, ICaL, IH",
            "",
            "comp = jx.Compartment()",
        ]

        # Map channel names to import sources
        builtin = {"Na", "K", "Leak"}
        custom = {"Kv3", "IM", "IAHP", "IT", "ICaL", "IH"}

        for ch in self.channels:
            if ch in builtin or ch in custom:
                lines.append(f"comp.insert({ch}())")
            else:
                lines.append(f"# Unknown channel: {ch}")

        lines.extend([
            f"",
            f"comp.set('radius', {self.radius})",
            f"comp.set('length', {self.length})",
            f"comp.set('capacitance', {self.capacitance})",
            f"comp.set('axial_resistivity', 100.0)",
        ])

        for param_name, config in self.param_config.items():
            lines.append(f"comp.set('{param_name}', {config.get('init', 0.001)})")

        return "\n".join(lines)

    def summary(self) -> str:
        return (
            f"Proposal #{self.proposal_id} (iter {self.iteration}): "
            f"channels=[{', '.join(self.channels)}] "
            f"loss={self.loss:.2f} "
            f"n_params={len(self.param_config)}"
        )


# ===========================================================================
# 2. Top-K Heap — maintains best proposals (SGA core data structure)
# ===========================================================================

class TopKHeap:
    """
    Min-heap of the K best model proposals, ranked by loss.
    This is the central data structure from SGA that persists across
    outer-loop iterations.
    """

    def __init__(self, k: int = 5):
        self.k = k
        self._heap: list[ModelProposal] = []
        self._counter = 0

    def push(self, proposal: ModelProposal):
        """Add a proposal. If heap is full, only keeps if better than worst."""
        if len(self._heap) < self.k:
            # Use negative loss for max-heap behavior (we want to evict worst)
            heapq.heappush(self._heap, proposal)
        elif proposal.loss < self.worst_loss():
            heapq.heapreplace(self._heap, proposal)

    def best(self) -> Optional[ModelProposal]:
        """Return the best (lowest loss) proposal."""
        if not self._heap:
            return None
        return min(self._heap, key=lambda p: p.loss)

    def worst_loss(self) -> float:
        """Return the worst (highest) loss in the heap."""
        if not self._heap:
            return float("inf")
        return max(p.loss for p in self._heap)

    def top_k(self) -> list[ModelProposal]:
        """Return all proposals sorted by loss (best first)."""
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
# 3. Diagnostic Report — translates inner-loop results for the LLM
# ===========================================================================

@dataclass
class DiagnosticReport:
    """
    Structured feedback from the inner loop that the LLM uses to
    revise the model structure. This is the "diagnostic-to-feedback
    translator" from the proposal.
    """
    proposal: ModelProposal
    specimen_id: int

    # Inner loop results
    final_loss: float = float("inf")
    n_sim_spikes: int = 0
    n_target_spikes: int = 0
    pearson_r: float = 0.0
    model_spikes: bool = False

    # Specific diagnostic flags (from proposal Section 3.3)
    no_spikes: bool = False
    wrong_firing_rate: bool = False
    wrong_adaptation: bool = False
    excessive_sag: bool = False
    broad_spikes: bool = False
    parameters_at_bounds: list = field(default_factory=list)

    def generate_feedback(self) -> str:
        """
        Generate natural-language feedback for the LLM, following the
        proposal's diagnostic categories.
        """
        lines = [
            f"## Inner Loop Results for Proposal #{self.proposal.proposal_id}",
            f"- Channels: {', '.join(self.proposal.channels)}",
            f"- Final MSE loss: {self.final_loss:.2f}",
            f"- Pearson correlation: {self.pearson_r:.3f}",
            f"- Simulated spikes: {self.n_sim_spikes}, Target spikes: {self.n_target_spikes}",
            "",
            "## Diagnostic Issues:",
        ]

        if self.no_spikes:
            lines.append(
                "- MODEL DOES NOT SPIKE: The simulation produced 0 spikes "
                "while the target has spikes. This suggests Na conductance is "
                "too low, the cell geometry gives insufficient current density, "
                "or a necessary depolarising current is missing."
            )

        if self.wrong_firing_rate and not self.no_spikes:
            rate_ratio = self.n_sim_spikes / max(self.n_target_spikes, 1)
            if rate_ratio < 0.5:
                lines.append(
                    f"- FIRING RATE TOO LOW: {self.n_sim_spikes} vs {self.n_target_spikes} "
                    "target spikes. Consider increasing Na conductance or adding "
                    "a fast-activating K+ channel (Kv3) that enables rapid repolarisation."
                )
            elif rate_ratio > 2.0:
                lines.append(
                    f"- FIRING RATE TOO HIGH: {self.n_sim_spikes} vs {self.n_target_spikes} "
                    "target spikes. Consider adding adaptation currents (I_M or I_AHP) "
                    "or increasing leak conductance."
                )

        if self.broad_spikes:
            lines.append(
                "- SPIKE WIDTH TOO BROAD: Consider adding Kv3-type fast delayed "
                "rectifier for rapid repolarisation (characteristic of PV+ FS neurons)."
            )

        if self.excessive_sag:
            lines.append(
                "- EXCESSIVE SAG during hyperpolarisation: I_h conductance may be "
                "too high. Consider removing IH or reducing its conductance."
            )

        if self.parameters_at_bounds:
            lines.append(
                f"- PARAMETERS AT BOUNDS: {', '.join(self.parameters_at_bounds)}. "
                "These parameters hit their optimisation limits, suggesting the "
                "model structure may need revision."
            )

        if not any([self.no_spikes, self.wrong_firing_rate,
                    self.broad_spikes, self.excessive_sag]):
            lines.append("- No major structural issues detected. "
                        "Consider fine-tuning or trying different channel combinations.")

        return "\n".join(lines)


# ===========================================================================
# 4. Prompt Templates — neuroscience-specific LLM prompts
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
- IM: Muscarinic M-type K+ (KCNQ/Kv7). Produces spike-frequency adaptation.
- IAHP: Ca²⁺-dependent K+ (medium AHP). Mediates afterhyperpolarisation.
- IT: T-type Ca²⁺ (low-voltage-activated). Rebound bursting in SST+ cells.
- ICaL: L-type Ca²⁺ (high-voltage-activated). Sustained calcium entry.
- IH: HCN (I_h). Sag and rebound on hyperpolarisation.

Always respond with a JSON object containing:
{
    "channels": ["Na", "K", "Leak", ...],
    "param_config": {
        "Na_gNa": {"init": 0.5, "lower": 0.05, "upper": 5.0},
        ...
    },
    "radius": 15.0,
    "capacitance": 1.0,
    "rationale": "Explanation of structural choices..."
}
"""


def make_initial_prompt(neuron_metadata: dict, ephys_features: dict) -> str:
    """
    Build the Stage 1 prompt: propose initial model structure given
    neuron metadata and electrophysiology features.
    """
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


def make_revision_prompt(diagnostic: DiagnosticReport,
                         heap_summary: str) -> str:
    """
    Build the Stage 2 outer-loop prompt: revise model based on
    inner-loop diagnostic feedback.
    """
    feedback = diagnostic.generate_feedback()

    return f"""## Task: Revise the model structure based on fitting results

{feedback}

## Current Best Proposals:
{heap_summary}

Based on the diagnostic feedback above, propose a REVISED model structure.
You may:
- Add channels that address identified issues
- Remove channels that are unnecessary
- Adjust parameter ranges and initial values
- Modify cell geometry

Explain your reasoning for each structural change.
Respond with a JSON object as specified in your instructions."""


# ===========================================================================
# 5. Outer Loop — orchestrates the bilevel optimisation
# ===========================================================================

class OuterLoop:
    """
    The SGA-style outer loop that orchestrates:
    1. LLM proposes model structure (or revises based on feedback)
    2. Inner loop (Jaxley gradient descent) optimises parameters
    3. Diagnostics generated from fitting results
    4. Results pushed to top-K heap
    5. LLM receives feedback and proposes revision

    Based on SGA's bilevel zigzag pattern.
    """

    def __init__(self, specimen_id: int, data_dir: str,
                 api_key: str = None, model: str = "claude-sonnet-4-20250514",
                 top_k: int = 5, provider: str = "anthropic",
                 inner_epochs: int = 80, inner_lr: float = 0.02):
        self.specimen_id = specimen_id
        self.data_dir = Path(data_dir)
        self.api_key = api_key
        self.model = model
        self.provider = provider
        self.heap = TopKHeap(k=top_k)
        self.history: list[dict] = []
        self.inner_epochs = inner_epochs
        self.inner_lr = inner_lr

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
        # Extract JSON from response (handle markdown code blocks)
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
            # Return a safe default
            data = {"channels": ["Na", "K", "Leak"], "rationale": "Parse failed"}

        return ModelProposal(
            proposal_id=self.heap.next_id(),
            iteration=iteration,
            parent_id=parent_id,
            channels=data.get("channels", ["Na", "K", "Leak"]),
            param_config=data.get("param_config", {}),
            radius=data.get("radius", 15.0),
            length=data.get("length", 31.4),
            capacitance=data.get("capacitance", 1.0),
            rationale=data.get("rationale", ""),
        )

    def _run_inner_loop(self, proposal: ModelProposal) -> DiagnosticReport:
        """
        Run the Jaxley inner loop for a proposal.

        Calls general_fit.fit_proposal() which dynamically builds a
        Jaxley cell from the proposal's channel list, runs gradient descent,
        and returns a DiagnosticReport with real fitting results.
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

        logger.info(f"Starting outer loop: max {max_iterations} iterations")

        for iteration in range(max_iterations):
            logger.info(f"\n{'='*60}")
            logger.info(f"OUTER LOOP ITERATION {iteration + 1}/{max_iterations}")
            logger.info(f"{'='*60}")

            # ---- Step 1: LLM proposes (or revises) model ----
            if iteration == 0:
                prompt = make_initial_prompt(neuron_metadata, ephys_features)
                parent_id = None
            else:
                best = self.heap.best()
                diagnostic = self._run_inner_loop(best)
                prompt = make_revision_prompt(diagnostic, self.heap.summary())
                parent_id = best.proposal_id

            logger.info("  Calling LLM for proposal...")
            try:
                response = self._call_llm(SYSTEM_PROMPT, prompt)
                proposal = self._parse_proposal(response, iteration, parent_id)
                logger.info(f"  LLM proposed: {proposal.channels}")
                logger.info(f"  Rationale: {proposal.rationale[:200]}")
            except Exception as e:
                logger.error(f"  LLM call failed: {e}")
                continue

            # ---- Step 2: Inner loop ----
            diagnostic = self._run_inner_loop(proposal)
            proposal.loss = diagnostic.final_loss
            proposal.diagnostics = asdict(diagnostic)

            # ---- Step 3: Push to heap ----
            self.heap.push(proposal)

            # ---- Log ----
            self.history.append({
                "iteration": iteration,
                "proposal": asdict(proposal),
                "diagnostic_feedback": diagnostic.generate_feedback(),
            })

            logger.info(f"\n{self.heap.summary()}")

            # ---- Early stopping: check if good enough ----
            best = self.heap.best()
            if best and best.diagnostics.get("model_spikes") and \
               best.diagnostics.get("firing_rate_error", 1.0) < 0.2:
                logger.info("  Fit is good enough — stopping early.")
                break

        # Save history
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

    # Test code generation
    p = ModelProposal(
        channels=["Na", "K", "Leak", "Kv3", "IM"],
        param_config={
            "Na_gNa": {"init": 0.5, "lower": 0.05, "upper": 5.0},
            "Kv3_gKv3": {"init": 0.01, "lower": 1e-4, "upper": 0.1},
        },
        radius=15.0,
        rationale="PV+ FS cell needs Kv3 for fast repolarisation",
    )
    print("\nGenerated Jaxley code:")
    print(p.to_jaxley_code())