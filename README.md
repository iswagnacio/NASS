

**Agentic Simulation of Cortical Neurons**

Research Proposal — Electrophysiological Response

**Agent Proposes  →  Agent Experiments  →  Agent Predicts**

LLM Agent discovers Hodgkin-Huxley-type ODE models

for cortical neuron types from Allen Cell Types Database recordings

March 2026  |  v3 (with implementation structure, benchmarks, and tool assembly)

# **1\. Core Idea**

This project uses an LLM agent to autonomously discover Hodgkin-Huxley-type ODE models for specific cortical neuron types. The workflow is a three-stage loop:

**Stage 1 — Propose:** The agent proposes a simulation equation for a target cortical neuron type. The equation describes how the neuron’s membrane potential evolves in response to stimulation. The agent draws on retrieved biological literature, ion channel expression data, and electrophysiology features to generate a plausible mathematical formulation.

**Stage 2 — Experiment & Optimize:** The agent takes the proposed equation, implements it as a differentiable simulation (Jaxley), and iteratively optimizes its parameters against experimental patch-clamp recordings. The agent evaluates the simulation output, diagnoses failures, and refines both the equation structure (outer loop) and its parameters (inner loop) in a bilevel optimization.

**Stage 3 — Predict:** The agent uses the validated, optimized equation to predict the neuron’s response to novel stimulation protocols not seen during optimization. This is the deliverable: a neuron-specific, interpretable ODE model that generalizes to novel inputs.

The final output is an HH-type model, but the novelty is that no human chose its structure — the agent discovered it. The model class is HH; the model selection process is what’s new.

# **2\. Positioning**

*Within the Li et al. (2025) survey taxonomy* ("LLMs Meet Virtual Cell"), our project is an LLM-as-Agent system that targets a new task (equation discovery), a new modality (electrophysiology), and a new cell type domain (cortical neurons). The survey organizes all LLM virtual cell work into two paradigms (Oracle vs Agent) and three core tasks (cellular representation, perturbation prediction, gene regulation). Our project extends this:

* **No surveyed agent discovers simulation equations.** CellForge designs neural network architectures; our agent discovers interpretable ODE equations.

* **Electrophysiology is entirely absent.** All three core tasks are defined in terms of gene expression (scRNA-seq). Membrane potential, firing patterns, and ion channel kinetics are not addressed.

* **No cortical neuron-specific work exists.** All perturbation benchmarks use cancer cell lines. Our focus on specific cortical neuron types directly addresses the survey’s identified gap in generalization to unseen cell types.

**Key prior work:** Jaxley (Deistler et al., Nature Methods 2025\) proves differentiable HH simulation works at scale but optimizes parameters within fixed human-designed structures. Gouwens et al. (Nature Comms 2018\) built models for 170 Allen neurons using genetic algorithms with human-designed templates. SGA (Ma et al., NeurIPS 2024\) demonstrated LLM-guided equation discovery for material science but has never been applied to neuroscience. Our project combines these: SGA’s bilevel equation-discovery loop \+ Jaxley’s differentiable neuron simulation \+ Allen’s standardized data.

**IMPLEMENTATION STRUCTURE**

## **3.1 System Input and Output**

**Input to the entire system:** A neuron type identifier (e.g., "PV+ fast-spiking interneuron, Layer 4, primary visual cortex") which selects a set of recordings from the Allen Cell Types Database via the Allen SDK.

**Output of the entire system:** A runnable HH-type ODE model — Python code defining the equations (which ion channels, what gating kinetics) \+ fitted parameter values — that can simulate that neuron type’s membrane voltage response to arbitrary current injection.

## **3.2 Stage 1: Agent Proposes Model Structure**

### **Agent Receives:**

* Neuron type metadata from Allen: cell type label, transgenic line, cortical layer, brain region, dendrite type (spiny/aspiny)

* Summary electrophysiology features: resting potential, input resistance, membrane time constant, rheobase, firing rate, adaptation index, upstroke:downstroke ratio, f-I curve slope

* Ion channel gene expression from Patch-seq (when available): which channel genes are transcriptomically expressed in this neuron type

* Retrieved literature context: published HH models for similar neuron types from ModelDB, known channel pharmacology from Channelpedia/ICGenealogy

### **Agent Outputs:**

* A Jaxley model definition — Python code specifying which ion channel mechanisms to include (e.g., cell.insert(Na()), cell.insert(Kdr()), cell.insert(Im()), cell.insert(HCN())), compartmental structure (single-compartment to start), and initial parameter ranges for each conductance

* A natural language rationale explaining structural choices (e.g., "PV+ interneurons use Kv3-family potassium channels for fast repolarization; I include Kv3 with fast deactivation kinetics and omit I\_h since these cells show minimal sag")

Implementation follows SGA’s outer-loop pattern: the LLM receives a structured prompt containing neuron metadata \+ features \+ retrieved knowledge, and generates code in a Jaxley-compatible template. The top-k heap from SGA maintains the best model proposals across iterations.

## **3.3 Stage 2: Bilevel Optimization**

### **Inner Loop — Jaxley Gradient Descent (automated, no agent involvement)**

**Input:** The model structure from Stage 1 \+ training sweeps from Allen (voltage traces from long square current injections at multiple amplitudes, typically 5–10 sweeps per neuron).

**Process:** Jaxley makes the model differentiable and runs gradient descent (Adam optimizer) to optimize conductance parameters (g\_Na, g\_K, g\_leak, etc.) and gating kinetics parameters. The loss function combines: (1) waveform MSE between simulated and recorded voltage traces, and (2) feature-based losses matching firing rate, spike height, spike width, and adaptation ratio (features Allen pre-computes for each neuron).

**Output:** Fitted parameter values \+ training loss \+ diagnostic metrics (does the model spike? firing rate match? spike shape match?).

Jaxley provides automatic differentiation through its implicit Euler ODE solver, GPU acceleration via JAX, and has been validated against Allen Cell Types Database recordings (Deistler et al. 2025, Extended Data Fig. 5). This replaces torchdiffeq as the inner-loop simulator — it is purpose-built for biophysical neuron models with built-in HH channel definitions.

### **Outer Loop — Agent Revises Structure**

The agent receives the inner loop results (loss value, diagnostic metrics, simulated vs. recorded trace comparison) and decides:

* If the fit is good enough → proceed to Stage 3

* If the model doesn’t spike at all → Na conductance range may be wrong, or a necessary current is missing

* If firing rate is correct but adaptation is wrong → add I\_M (muscarinic) or I\_AHP (calcium-dependent afterhyperpolarization) current

* If there is excessive sag during hyperpolarization → I\_h conductance is too high, or should be removed entirely

* If spike width is too broad → add Kv3-type fast delayed rectifier

The agent outputs a revised model structure (modified Jaxley code), and the inner loop runs again. This follows the SGA bilevel zigzag pattern. Based on SGA’s experiments in material science, convergence typically requires 3–8 outer iterations.

## **3.4 Stage 3: Predict Novel Stimuli**

**Input:** The converged model from Stage 2 \+ held-out stimulus protocols from Allen that were NOT used during fitting.

**Process:** Run the fitted model forward on held-out stimuli (noise injection, ramp current, short square pulses) and compare predicted voltage traces to actual recorded traces.

**Output:** Predicted membrane voltage traces \+ evaluation metrics on held-out data (see Benchmark Design below).

## **3.5 Data Flow Diagram**

| Stage | Input | Process | Output |
| :---- | :---- | :---- | :---- |
| 1: Propose | Neuron metadata, ephys features, Patch-seq channels, literature | LLM generates Jaxley model code via SGA outer-loop prompting | Model structure (Python code) \+ rationale |
| 2: Inner | Model code \+ training sweeps (long square, 5–10 traces) | Jaxley gradient descent on conductance \+ gating parameters | Fitted parameters \+ loss \+ diagnostics |
| 2: Outer | Inner loop results \+ diagnostics | LLM evaluates fit, diagnoses issues, revises model structure | Revised model code (repeat inner loop) |
| 3: Predict | Converged model \+ held-out stimuli (noise, ramp, short square) | Forward simulation on novel protocols | Predicted traces \+ generalization metrics |

## **3.6 Codebase Structure**

The implementation would be organized as follows:

| Module | Responsibility | Key Dependencies |
| :---- | :---- | :---- |
| data/ | Allen SDK interface: download NWB files, extract sweeps by stimulus type, split into train/held-out, extract summary features | AllenSDK, pynwb, h5py |
| agent/ | SGA-style outer loop: LLM prompt templates for proposing model structure, parsing agent output into Jaxley code, feedback prompts for revision, top-k heap of best proposals | OpenAI/Anthropic API, SGA codebase |
| simulation/ | Jaxley interface: compile agent-generated model code, run inner-loop gradient optimization against training sweeps, return loss \+ diagnostics | Jaxley, JAX, optax |
| channels/ | Custom ion channel library: Kv3, I\_M, I\_AHP, I\_h, I\_T, I\_CaL beyond Jaxley’s built-in HH. Agent can reference these in its proposals | Jaxley channel API |
| evaluation/ | Compute held-out metrics: spike time coincidence, firing rate error, f-I curve prediction, subthreshold variance explained, spike shape metrics | numpy, scipy |
| baselines/ | Comparison models: LIF, Izhikevich, AdEx, fixed-HH-Jaxley, Gouwens et al. (2018) Allen model loader | Jaxley, NEURON (optional) |

# **4\. Benchmark Design**

There is no established benchmark for LLM-guided neuron model discovery. We construct one from the Allen Cell Types Database, leveraging its standardized stimulus battery:

## **4.1 Primary Benchmark: Cross-Stimulus Generalization**

The Allen protocol applies a standardized battery of stimuli to each neuron. We split them into training and held-out sets:

| Split | Stimulus Types | Purpose |
| :---- | :---- | :---- |
| Training (Stage 2\) | Long square current injections at 3–5 amplitudes spanning sub-threshold to 2× rheobase | Captures basic firing properties: threshold, f-I relationship, adaptation, spike shape |
| Held-out Set 1 | Noise 1 & Noise 2 protocols (fluctuating current mimicking synaptic input) | Tests response to realistic, time-varying input — most demanding generalization test |
| Held-out Set 2 | Ramp current injection (linear increase over \~1 second) | Tests threshold detection and recruitment dynamics |
| Held-out Set 3 | Short square pulses (2–3 ms) | Tests single-spike dynamics and rheobase precision |
| Held-out Set 4 | Long square amplitudes not used in training | Tests f-I curve interpolation/extrapolation |

## **4.2 Evaluation Metrics**

| Metric | What It Measures | Standard In |
| :---- | :---- | :---- |
| Spike time coincidence factor (Γ) | Do spikes occur at the right times? Penalizes missing/extra spikes with temporal tolerance window (\~4 ms) | Computational neuroscience (Kistler et al. 1997\) |
| Firing rate error (relative) | |predicted\_rate − true\_rate| / true\_rate across held-out amplitudes | Allen Institute model validation |
| f-I curve RMSE | Root mean squared error of predicted vs. actual firing rate across current amplitudes | Standard electrophysiology |
| Subthreshold variance explained (R²) | Does model capture passive membrane dynamics (non-spiking voltage fluctuations)? | Jaxley paper (Deistler et al. 2025\) |
| Spike shape error | Mean squared error of average spike waveform (peak, trough, half-width, upstroke:downstroke ratio) | Allen Cell Types feature set |
| Model complexity | Number of ion currents \+ number of free parameters in the discovered model | Parsimony measure |

## **4.3 Comparison Baselines**

| Baseline | What It Tests | Source |
| :---- | :---- | :---- |
| Gouwens et al. (2018) Allen models | Human-designed structure \+ genetic algorithm optimization on same Allen data. The direct comparison: does agent-guided discovery match or beat expert-guided modeling? | Allen Institute (170 neurons, published models) |
| Jaxley fixed-HH (Na+K+leak only) | Ablation: same Jaxley gradient optimization but with a generic 3-channel model. Shows whether the agent’s structural choices actually help | Our implementation |
| Jaxley fixed-HH (rich template) | Ablation: Jaxley with all available channels inserted by default, agent not involved. Shows whether intelligent channel selection matters vs. brute-force inclusion | Our implementation |
| Leaky Integrate-and-Fire (LIF) | The simplest possible model — no ion channels, just threshold \+ reset. Lower bound on performance | Allen SDK built-in |
| Izhikevich model | Phenomenological model with 4 parameters. Fast to fit but no biophysical interpretability | Standard implementation |
| AdEx (Adaptive Exponential IF) | 2-variable model with adaptation. Good balance of speed and accuracy for simple neurons | Standard implementation |

## **4.4 Secondary Benchmark: Cross-Neuron Generalization**

Within a neuron type (e.g., PV+ fast-spiking, \~20–50 neurons in Allen), use some neurons for Stage 1–2 discovery and parameter fitting, hold out others entirely. Test whether the model structure the agent discovered for one PV+ neuron generalizes to other PV+ neurons with only re-fitting of parameters (i.e., skip Stage 1, run only Stage 2 inner loop on the held-out neuron). This tests whether the agent discovers a true type-level model, not just an individual-neuron overfit.

## **4.5 Test Neuron Types (Recommended Progression)**

| Phase | Neuron Type | Why Start Here | Expected Difficulty |
| :---- | :---- | :---- | :---- |
| Phase 1 | PV+ fast-spiking interneuron | Well-characterized Kv3 channels, narrow spikes, minimal adaptation, clean electrophysiological signature. Most Allen data available. | Low — relatively few ion currents needed |
| Phase 2 | SST+ low-threshold-spiking | Known I\_T (T-type calcium) current for rebound bursting. Tests whether agent discovers calcium channels | Medium — requires calcium dynamics |
| Phase 3 | Layer 5 thick-tufted pyramidal | Regular spiking with adaptation, I\_M and I\_AHP currents, complex dendritic computation. May need multi-compartment | High — many interacting currents, potential multi-compartment |
| Phase 4 | VIP+ irregular-spiking | Highly variable firing patterns, less characterized channels. Tests agent on underspecified problems | High — limited prior knowledge, variable phenotype |

# **5\. Tool Assembly**

No single existing tool covers all three stages. We assemble the pipeline from proven open-source components:

| Tool | Role in Pipeline | License | Repo |
| :---- | :---- | :---- | :---- |
| Jaxley | Stage 2 inner loop: differentiable HH simulation on GPU with automatic differentiation through ODE solver. Purpose-built for biophysical neuron models. Replaces generic torchdiffeq. | Apache 2.0 | github.com/jaxleyverse/jaxley |
| SGA codebase | Stage 1–2 outer loop scaffolding: top-k heap, exploit/explore offspring, LLM-to-simulation interface, feedback loop pattern. Adapt Physics(nn.Module) template to Jaxley NeuronModel. | MIT | github.com/PingchuanMa/SGA |
| OpenHands SDK | Agent runtime backbone: sandboxed code execution, self-debugging, error recovery. CellForge uses this internally. | MIT | github.com/OpenHands/OpenHands |
| Allen SDK | Data access: download NWB files, extract sweeps, load pre-computed electrophysiology features, access morphology reconstructions. | BSD 3-Clause | github.com/AllenInstitute/AllenSDK |
| CellForge (reference) | Architecture reference for multi-agent design patterns: Task Analysis, Method Design, graph-based expert discussion. We adapt the agent collaboration pattern, not the perturbation prediction code. | Open source | github.com/gersteinlab/CellForge |
| ModelDB / ICGenealogy | Stage 1 knowledge retrieval: existing published HH models for similar neuron types as initialization/reference for agent proposals. | Open access | modeldb.yale.edu / icg.neurotheory.ox.ac.uk |

### **What We Build Custom (Not Available Off-the-Shelf)**

* The neuron-specific prompt templates that guide the LLM to propose HH equations (adapting SGA’s material-science prompts to neuroscience vocabulary)

* A custom ion channel library for Jaxley extending its built-in channels (Kv3, I\_M, I\_AHP, I\_T, I\_CaL, I\_h with realistic cortical neuron kinetics)

* The evaluation pipeline comparing agent-discovered models against Allen features and Gouwens et al. baselines

* The diagnostic-to-feedback translator that converts Jaxley optimization results into structured LLM prompts for the outer loop

**Key insight:** SGA gives us \~70% of the infrastructure. We replace its material-science simulation backend with Jaxley’s neuron ODE solver, replace its constitutive-law templates with HH equation templates, and point it at Allen patch-clamp data instead of material deformation data. The LLM-loop scaffolding (top-k heap, exploit/explore, bilevel feedback) is reusable as-is.

# **6\. Timeline & Milestones**

| Week | Milestone | Deliverable |
| :---- | :---- | :---- |
| 1–2 | Data pipeline & environment setup | Allen SDK configured; NWB download and sweep extraction scripts for PV+ fast-spiking interneurons; train/held-out splits defined; electrophysiology feature loading verified for 20–50 PV+ neurons |
| 3–4 | Jaxley baseline & inner loop | Fixed 3-channel HH model (Na\+K\+leak) fitting a single PV+ neuron via Jaxley gradient descent on GPU; inner loop converging; diagnostic outputs operational (spiking check, firing rate, spike shape, simulated vs. recorded overlay) |
| 5–6 | Evaluation pipeline & comparison baselines | All held-out metrics implemented (spike time coincidence, firing rate error, f-I RMSE, subthreshold R², spike shape error); LIF, Izhikevich, AdEx baselines producing reference numbers; Gouwens et al. models loaded if available |
| 7–8 | Custom channel library & SGA scaffolding | Extended Jaxley channels built (Kv3, I\_M, I\_AHP, I\_T, I\_CaL, I\_h with cortical kinetics); SGA codebase forked and adapted — top-k heap, exploit/explore offspring, Jaxley backend replacing material-science simulator |
| 9–10 | Agent prompt engineering & Stage 1 prototype | Neuron-specific LLM prompt templates ingesting metadata, ephys features, Patch-seq expression, and ModelDB/ICGenealogy literature; agent generating syntactically valid Jaxley code with biologically sensible channel selections; diagnostic-to-feedback translator built |
| 11–12 | First end-to-end runs & preliminary evaluation | Full three-stage pipeline connected: agent proposes (Stage 1), inner loop optimizes (Stage 2 inner), diagnostics feed back, agent revises (Stage 2 outer); outer loop converging within 3–8 iterations on PV+ neurons; agent-discovered models evaluated on held-out stimuli and compared against all baselines; preliminary quantitative results documented |

**Scope Note:** This 3-month timeline focuses on PV+ fast-spiking interneurons (Phase 1 neuron type) and delivers a working end-to-end prototype with initial benchmark results. Scaling to additional neuron types (SST+, L5 pyramidal, VIP+), systematic cross-neuron generalization testing, and paper preparation are deferred to subsequent months.

# **7\. Key References**

### **Core Method Papers**

* SGA (Ma et al., NeurIPS 2024\) — Bilevel optimization: LLM proposes equations, differentiable simulation optimizes parameters. github.com/PingchuanMa/SGA

* Jaxley (Deistler et al., Nature Methods 2025\) — Differentiable biophysical neuron simulation with GPU acceleration. github.com/jaxleyverse/jaxley

* CellForge (Tang et al., 2026\) — Multi-agent virtual cell model design. github.com/gersteinlab/CellForge

* Li et al. (2025) — "LLMs Meet Virtual Cell" survey. Taxonomy and field positioning.

### **Neuron Modeling Papers**

* Gouwens et al. (Nature Comms 2018\) — Systematic biophysical models for 170 Allen cortical neurons via genetic algorithm. Primary comparison baseline.

* HHBPTT (bioRxiv 2025\) — Backpropagation-through-time for HH conductance estimation. github.com/skysky2333/HHBPTT

* Pospischil et al. (Biol Cybernetics 2008\) — Minimal HH models for cortical/thalamic neuron classes.

* Dura-Bernal et al. (J Physiol 2024\) — Review of cortical network simulations with conductance-based models.

### **Virtual Cell / Perturbation Papers**

* VCWorld (Wei et al., 2025\) — LLM \+ knowledge graph for perturbation prediction. github.com/GENTEL-lab/VCWorld

* STATE (Adduri et al., 2025\) — 170M-cell perturbation foundation model. github.com/ArcInstitute/state

* AIVC Roadmap (Bunne et al., Cell 2024\) — Multi-scale virtual cell priorities.

* BioDiscoveryAgent (Roohani et al., 2024\) — Full-stack closed-loop research agent.

* STELLA (Jin et al., 2025\) — Self-evolving agent with Template Library \+ Tool Ocean.

### **Data Sources**

* Allen Cell Types Database — celltypes.brain-map.org — \~2000 neurons with standardized patch-clamp \+ morphology

* Patch-seq (Allen) — portal.brain-map.org — \~4000 neurons with joint electrophysiology \+ transcriptomics

* ModelDB — modeldb.yale.edu — \~500 published cortical HH models

* ICGenealogy — icg.neurotheory.ox.ac.uk — \~2500 curated ion channel models

* Channelpedia — channelpedia.epfl.ch — Ion channel parameter database (BBP)