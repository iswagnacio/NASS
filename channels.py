"""
Custom Ion Channel Library for Cortical Neurons (Weeks 7–8, Part 1)
===================================================================

Extends Jaxley's built-in Na/K/Leak with channels critical for cortical
neuron modeling. These are the channels the LLM agent can reference
when proposing model structures.

Channels implemented (from NASS proposal Section 5):
    - Kv3:   Fast delayed rectifier — enables high-frequency firing in PV+ FS cells
    - IM:    Muscarinic (M-type) K+ — spike frequency adaptation
    - IAHP:  Ca²⁺-dependent K+ (afterhyperpolarization) — adaptation in pyramidals
    - IT:    T-type Ca²⁺ — low-threshold bursting in SST+ cells
    - ICaL:  L-type Ca²⁺ (high-voltage-activated) — sustained depolarisation
    - IH:    Hyperpolarization-activated cation (HCN) — sag and rebound

Kinetics are based on Pospischil et al. (2008), Biol Cybernetics, which
provides minimal HH-type models for cortical/thalamic neuron classes.
This is the primary reference the proposal cites for cortical HH channel kinetics.

All channels follow Jaxley's Channel API:
    - channel_params: dict of {name: default_value}
    - channel_states: dict of {name: default_value}
    - current_name: string identifying the current (e.g. "i_Kv3")
    - update_states(states, dt, v, params) -> dict of updated states
    - compute_current(states, v, params) -> current in mA/cm²
    - init_state(v, params) -> dict of steady-state values

Units follow Jaxley/NEURON convention:
    - Voltage: mV
    - Conductance density: S/cm² (params); current returned in mA/cm²
    - Current: returned as mA/cm² (Jaxley convention)
    - Time: ms

Usage:
    from channels import Kv3, IM, IAHP, IT, ICaL, IH
    comp = jx.Compartment()
    comp.insert(Kv3())
    comp.insert(IM())
    # etc.
"""

import jax.numpy as jnp
from jaxley.channels import Channel
from jaxley.solver_gate import solve_gate_exponential


# ---------------------------------------------------------------------------
# Helper: safe exponential to avoid overflow
# ---------------------------------------------------------------------------

def _safe_exp(x):
    """Exponential clipped to avoid overflow."""
    return jnp.exp(jnp.clip(x, -50.0, 50.0))


def _sigmoid(v, v_half, k):
    """Boltzmann sigmoid: 1 / (1 + exp(-(v - v_half) / k))."""
    return 1.0 / (1.0 + _safe_exp(-(v - v_half) / k))


def _alpha_beta_from_inf_tau(x_inf, tau):
    """Convert steady-state and time constant to alpha/beta rates."""
    alpha = x_inf / jnp.maximum(tau, 0.01)
    beta = (1.0 - x_inf) / jnp.maximum(tau, 0.01)
    return alpha, beta


# ===========================================================================
# Kv3 — Fast Delayed Rectifier K+ Channel
# ===========================================================================

class Kv3(Channel):
    """
    Kv3 (Shaw-related) fast delayed rectifier potassium channel.
 
    Critical for PV+ fast-spiking interneurons: enables rapid repolarisation
    and high-frequency firing. Key features:
        - High activation threshold (~-10 mV)
        - Very fast deactivation (τ ~ 1-2 ms)
        - No inactivation (or very slow)
 
    Kinetics from Erisir et al. (1999) J Neurophysiol, adapted to the
    Pospischil et al. (2008) framework.
 
    I_Kv3 = g_Kv3 * n² * (V - E_K)
 
    GRADIENT SAFETY NOTE (March 2026):
        - tau_n uses softabs (sqrt(x² + 1)) instead of jnp.abs to avoid
          the non-differentiable kink at V = -16 mV. During a spike, V
          crosses -16 mV on every upstroke and downstroke; 200+ spikes
          means 400+ gradient discontinuities compounding through BPTT.
        - tau_n is clipped to [0.1, 50.0] ms to prevent extreme rates
          that can overflow float32 Jacobians.
    """
 
    def __init__(self, name=None):
        self.current_is_in_mA_per_cm2 = True
        super().__init__(name)
        self.channel_params = {
            f"{self._name}_gKv3": 1e-3,   # S/cm² (default, will be optimized)
            "eK": -90.0,                   # shared global K+ reversal (mV)
        }
        self.channel_states = {
            f"{self._name}_n": 0.0,
        }
        self.current_name = "i_Kv3"
 
    def update_states(self, states, dt, v, params):
        n = states[f"{self._name}_n"]
 
        # Steady state — Boltzmann sigmoid (unchanged)
        n_inf = _sigmoid(v, -12.4, 6.8)
 
        # Time constant — bell curve centered at V = -16 mV
        # ORIGINAL (gradient-unsafe):
        #   tau_n = 0.25 + 4.35 * _safe_exp(-jnp.abs(v + 16.0) / 40.0)
        # FIXED: softabs avoids the non-differentiable kink at V = -16
        soft_abs = jnp.sqrt((v + 16.0) ** 2 + 1.0)
        tau_n = 0.25 + 4.35 * _safe_exp(-soft_abs / 40.0)
 
        # Clamp tau to prevent extreme rates
        tau_n = jnp.clip(tau_n, 0.1, 50.0)
 
        alpha, beta = _alpha_beta_from_inf_tau(n_inf, tau_n)
        new_n = solve_gate_exponential(n, dt, alpha, beta)
        return {f"{self._name}_n": new_n}
 
    def compute_current(self, states, v, params):
        n = states[f"{self._name}_n"]
        g = params[f"{self._name}_gKv3"] * n ** 2
        return g * (v - params["eK"])
 
    def init_state(self, v, params):
        n_inf = _sigmoid(v, -12.4, 6.8)
        return {f"{self._name}_n": n_inf}


# ===========================================================================
# IM — Muscarinic (M-type) K+ Channel
# ===========================================================================

class IM(Channel):
    """
    Muscarinic (M-type) potassium channel (Kv7 / KCNQ).

    Slow, non-inactivating K+ current that activates at subthreshold voltages.
    Responsible for spike-frequency adaptation and regulation of excitability.
    Important for regular-spiking pyramidal neurons.

    Kinetics from Pospischil et al. (2008), based on Adams et al. (1982).

    I_M = g_M * p * (V - E_K)
    """

    def __init__(self, name=None):
        self.current_is_in_mA_per_cm2 = True
        super().__init__(name)
        self.channel_params = {
            f"{self._name}_gM": 1e-5,     # S/cm² (typically very small)
            "eK": -90.0,                   # shared global K+ reversal (mV)
        }
        self.channel_states = {
            f"{self._name}_p": 0.0,
        }
        self.current_name = "i_M"

    def update_states(self, states, dt, v, params):
        p = states[f"{self._name}_p"]

        p_inf = _sigmoid(v, -35.0, 10.0)
        tau_p = 1000.0 / (3.3 * (_safe_exp((v + 35.0) / 20.0) +
                                  _safe_exp(-(v + 35.0) / 20.0)))
        # Clamp tau to reasonable range
        tau_p = jnp.clip(tau_p, 10.0, 1000.0)

        alpha, beta = _alpha_beta_from_inf_tau(p_inf, tau_p)
        new_p = solve_gate_exponential(p, dt, alpha, beta)
        return {f"{self._name}_p": new_p}

    def compute_current(self, states, v, params):
        p = states[f"{self._name}_p"]
        g = params[f"{self._name}_gM"] * p
        return g * (v - params["eK"])

    def init_state(self, v, params):
        p_inf = _sigmoid(v, -35.0, 10.0)
        return {f"{self._name}_p": p_inf}


# ===========================================================================
# IAHP — Ca²⁺-dependent K+ Channel (Afterhyperpolarization)
# ===========================================================================

class IAHP(Channel):
    """
    Calcium-dependent potassium channel (medium AHP).

    Mediates the medium afterhyperpolarization following spike trains.
    Important for adaptation in pyramidal neurons.

    Simplified model: uses voltage as a proxy for calcium concentration
    (since we don't track [Ca²⁺]_i in the basic pipeline). The activation
    variable q tracks a slow, voltage-dependent process that mimics
    calcium-driven activation.

    Kinetics loosely based on Pospischil et al. (2008).

    I_AHP = g_AHP * q * (V - E_K)
    """

    def __init__(self, name=None):
        self.current_is_in_mA_per_cm2 = True
        super().__init__(name)
        self.channel_params = {
            f"{self._name}_gAHP": 1e-5,
            "eK": -90.0,                   # shared global K+ reversal (mV)
        }
        self.channel_states = {
            f"{self._name}_q": 0.0,
        }
        self.current_name = "i_AHP"

    def update_states(self, states, dt, v, params):
        q = states[f"{self._name}_q"]

        # Activation depends on depolarisation level (proxy for Ca²⁺ entry)
        q_inf = _sigmoid(v, -20.0, 5.0)
        tau_q = 100.0 + 500.0 / (1.0 + _safe_exp((v + 40.0) / 5.0))

        alpha, beta = _alpha_beta_from_inf_tau(q_inf, tau_q)
        new_q = solve_gate_exponential(q, dt, alpha, beta)
        return {f"{self._name}_q": new_q}

    def compute_current(self, states, v, params):
        q = states[f"{self._name}_q"]
        g = params[f"{self._name}_gAHP"] * q
        return g * (v - params["eK"])

    def init_state(self, v, params):
        q_inf = _sigmoid(v, -20.0, 5.0)
        return {f"{self._name}_q": q_inf}


# ===========================================================================
# IT — T-type Ca²⁺ Channel (Low-Voltage-Activated)
# ===========================================================================

class IT(Channel):
    """
    T-type (low-voltage-activated) calcium channel.

    Produces rebound bursting in SST+ low-threshold-spiking interneurons.
    Activates at subthreshold voltages, inactivates rapidly.
    Key test for Phase 2 neurons in the proposal.

    Kinetics from Pospischil et al. (2008), based on Huguenard & McCormick (1992).

    I_T = g_T * m²_T * h_T * (V - E_Ca)
    """

    def __init__(self, name=None):
        self.current_is_in_mA_per_cm2 = True
        super().__init__(name)
        self.channel_params = {
            f"{self._name}_gT": 1e-4,
        }
        self.channel_states = {
            f"{self._name}_m": 0.0,
            f"{self._name}_h": 1.0,
        }
        self.current_name = "i_CaT"

    def update_states(self, states, dt, v, params):
        m = states[f"{self._name}_m"]
        h = states[f"{self._name}_h"]

        # Activation
        m_inf = _sigmoid(v, -57.0, 6.2)
        tau_m = 0.13 + 3.68 / (1.0 + _safe_exp((v + 27.0) / (-10.0))) + \
                3.68 / (1.0 + _safe_exp(-(v + 102.0) / 15.0))
        tau_m = jnp.maximum(tau_m, 0.1)

        # Inactivation (slow)
        h_inf = _sigmoid(v, -81.0, -4.0)
        tau_h = 8.2 + (56.6 + 0.27 * _safe_exp((v + 48.0) / 4.0)) / \
                (1.0 + _safe_exp((v + 22.0) / (-5.0)))
        tau_h = jnp.maximum(tau_h, 0.5)

        alpha_m, beta_m = _alpha_beta_from_inf_tau(m_inf, tau_m)
        alpha_h, beta_h = _alpha_beta_from_inf_tau(h_inf, tau_h)

        new_m = solve_gate_exponential(m, dt, alpha_m, beta_m)
        new_h = solve_gate_exponential(h, dt, alpha_h, beta_h)

        return {f"{self._name}_m": new_m, f"{self._name}_h": new_h}

    def compute_current(self, states, v, params):
        m = states[f"{self._name}_m"]
        h = states[f"{self._name}_h"]
        g = params[f"{self._name}_gT"] * m ** 2 * h
        e_ca = 120.0
        return g * (v - e_ca)

    def init_state(self, v, params):
        m_inf = _sigmoid(v, -57.0, 6.2)
        h_inf = _sigmoid(v, -81.0, -4.0)
        return {f"{self._name}_m": m_inf, f"{self._name}_h": h_inf}


# ===========================================================================
# ICaL — L-type Ca²⁺ Channel (High-Voltage-Activated)
# ===========================================================================

class ICaL(Channel):
    """
    L-type (high-voltage-activated) calcium channel.

    Sustained Ca²⁺ entry during depolarisation. Important for dendritic
    calcium spikes in L5 pyramidal neurons and for driving Ca²⁺-dependent
    K+ channels.

    Kinetics based on Reuveni et al. (1993) / Pospischil et al. (2008).

    I_CaL = g_CaL * m²_CaL * (V - E_Ca)
    """

    def __init__(self, name=None):
        self.current_is_in_mA_per_cm2 = True
        super().__init__(name)
        self.channel_params = {
            f"{self._name}_gCaL": 1e-4,
        }
        self.channel_states = {
            f"{self._name}_m": 0.0,
        }
        self.current_name = "i_CaL"

    def update_states(self, states, dt, v, params):
        m = states[f"{self._name}_m"]

        m_inf = _sigmoid(v, -10.0, 6.5)
        # L-type has relatively slow kinetics
        tau_m = 1.5 + 10.0 / (1.0 + _safe_exp((v + 25.0) / 10.0))
        tau_m = jnp.maximum(tau_m, 0.5)

        alpha, beta = _alpha_beta_from_inf_tau(m_inf, tau_m)
        new_m = solve_gate_exponential(m, dt, alpha, beta)
        return {f"{self._name}_m": new_m}

    def compute_current(self, states, v, params):
        m = states[f"{self._name}_m"]
        g = params[f"{self._name}_gCaL"] * m ** 2
        e_ca = 120.0
        return g * (v - e_ca)

    def init_state(self, v, params):
        m_inf = _sigmoid(v, -10.0, 6.5)
        return {f"{self._name}_m": m_inf}


# ===========================================================================
# IH — Hyperpolarization-Activated Cation Channel (HCN)
# ===========================================================================

class IH(Channel):
    """
    Hyperpolarization-activated cation current (I_h / HCN).

    Mixed Na+/K+ channel that activates on hyperpolarisation. Produces
    the characteristic "sag" during hyperpolarising current injection.
    Present in pyramidal neurons and some interneurons.

    Kinetics from Pospischil et al. (2008), based on McCormick & Pape (1990).

    I_H = g_H * r * (V - E_H)
    where E_H ≈ -40 mV (mixed cation reversal)
    """

    def __init__(self, name=None):
        self.current_is_in_mA_per_cm2 = True
        super().__init__(name)
        self.channel_params = {
            f"{self._name}_gH": 1e-5,
        }
        self.channel_states = {
            f"{self._name}_r": 0.0,
        }
        self.current_name = "i_H"

    def update_states(self, states, dt, v, params):
        r = states[f"{self._name}_r"]

        # Activates on hyperpolarisation (note negative slope)
        r_inf = _sigmoid(v, -80.0, -10.0)
        tau_r = 100.0 + 1000.0 / (_safe_exp((v + 70.0) / 20.0) +
                                    _safe_exp(-(v + 70.0) / 20.0))
        tau_r = jnp.clip(tau_r, 10.0, 5000.0)

        alpha, beta = _alpha_beta_from_inf_tau(r_inf, tau_r)
        new_r = solve_gate_exponential(r, dt, alpha, beta)
        return {f"{self._name}_r": new_r}

    def compute_current(self, states, v, params):
        r = states[f"{self._name}_r"]
        g = params[f"{self._name}_gH"] * r
        e_h = -40.0  # Mixed cation reversal
        return g * (v - e_h)

    def init_state(self, v, params):
        r_inf = _sigmoid(v, -80.0, -10.0)
        return {f"{self._name}_r": r_inf}


# ===========================================================================
# Channel Registry — for agent to look up available channels
# ===========================================================================

CHANNEL_REGISTRY = {
    "Kv3": {
        "class": Kv3,
        "ion": "K",
        "description": "Fast delayed rectifier (Kv3/Shaw). Enables high-frequency "
                       "firing in PV+ FS interneurons via rapid repolarisation.",
        "key_param": "gKv3",
        "typical_range": (1e-4, 0.1),
        "neuron_types": ["PV+ fast-spiking", "chandelier"],
    },
    "IM": {
        "class": IM,
        "ion": "K",
        "description": "Muscarinic M-type (KCNQ/Kv7). Slow non-inactivating K+ "
                       "current producing spike-frequency adaptation.",
        "key_param": "gM",
        "typical_range": (1e-6, 1e-3),
        "neuron_types": ["L5 pyramidal", "regular-spiking"],
    },
    "IAHP": {
        "class": IAHP,
        "ion": "K",
        "description": "Ca²⁺-dependent K+ (medium AHP). Mediates medium "
                       "afterhyperpolarization and adaptation.",
        "key_param": "gAHP",
        "typical_range": (1e-6, 1e-3),
        "neuron_types": ["L5 pyramidal", "regular-spiking"],
    },
    "IT": {
        "class": IT,
        "ion": "Ca",
        "description": "T-type Ca²⁺ (low-voltage-activated). Produces rebound "
                       "bursting in SST+ LTS interneurons.",
        "key_param": "gT",
        "typical_range": (1e-5, 1e-2),
        "neuron_types": ["SST+ low-threshold-spiking", "thalamic relay"],
    },
    "ICaL": {
        "class": ICaL,
        "ion": "Ca",
        "description": "L-type Ca²⁺ (high-voltage-activated). Sustained calcium "
                       "entry during depolarisation.",
        "key_param": "gCaL",
        "typical_range": (1e-5, 1e-2),
        "neuron_types": ["L5 pyramidal", "thalamic"],
    },
    "IH": {
        "class": IH,
        "ion": "mixed",
        "description": "HCN (I_h). Hyperpolarisation-activated cation current "
                       "producing sag and rebound depolarisation.",
        "key_param": "gH",
        "typical_range": (1e-6, 1e-3),
        "neuron_types": ["L5 pyramidal", "thalamic relay"],
    },
}


def list_channels() -> str:
    """Pretty-print all available channels for the agent."""
    lines = ["Available custom ion channels:"]
    for name, info in CHANNEL_REGISTRY.items():
        lines.append(f"\n  {name} ({info['ion']}+)")
        lines.append(f"    {info['description']}")
        lines.append(f"    Key param: {info['key_param']} "
                     f"(range: {info['typical_range'][0]:.0e}–{info['typical_range'][1]:.0e} S/cm²)")
        lines.append(f"    Neuron types: {', '.join(info['neuron_types'])}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(list_channels())

    # Quick smoke test: build a PV+ FS cell with Kv3
    import jaxley as jx
    from jaxley.channels import Na, K, Leak

    comp = jx.Compartment()
    comp.insert(Na())
    comp.insert(K())
    comp.insert(Leak())
    comp.insert(Kv3())
    comp.insert(IM())

    print(f"\nNode columns after inserting Na+K+Leak+Kv3+IM:")
    for col in sorted(comp.nodes.columns):
        if any(ch in col for ch in ["Na", "K", "Leak", "Kv3", "IM", "gK", "gN", "gM"]):
            print(f"  {col}")

    print("\nSmoke test passed!")