"""
Custom Ion Channel Library for Cortical Neurons (Weeks 7-8, Part 1)
===================================================================

Extends BrainPy's built-in INa_HH1952/IK_HH1952/IL with channels critical
for cortical neuron modeling. These are the channels the LLM agent can
reference when proposing model structures.

Channels implemented (from NASS proposal Section 5):
    - Kv3:   Fast delayed rectifier -- enables high-frequency firing in PV+ FS cells
    - IM:    Muscarinic (M-type) K+ -- spike frequency adaptation
    - IAHP:  Ca2+-dependent K+ (afterhyperpolarization) -- adaptation in pyramidals
    - IT:    T-type Ca2+ -- low-threshold bursting in SST+ cells
    - ICaL:  L-type Ca2+ (high-voltage-activated) -- sustained depolarisation
    - IH:    Hyperpolarization-activated cation (HCN) -- sag and rebound

Kinetics are based on Pospischil et al. (2008), Biol Cybernetics, which
provides minimal HH-type models for cortical/thalamic neuron classes.

All channels follow BrainPy's IonChannel API:
    - master_type = bp.dyn.CondNeuGroup  (receives V from parent neuron)
    - __init__(size, ...):  declare parameters and gating Variable(s)
    - reset_state(V, batch_size):  initialise gating variables to steady-state
    - update(V):  advance gating variables one timestep via ODE integrator
    - current(V):  return ionic current density

Units follow Jaxley/NEURON convention (preserved from original):
    - Voltage: mV
    - Conductance density: S/cm2 (params); current returned in mA/cm2
    - Time: ms

Usage (BrainPy equivalent of the original Jaxley usage):
    # --- Original Jaxley ---
    # comp = jx.Compartment()
    # comp.insert(Kv3())
    # comp.insert(IM())
    #
    # --- BrainPy ---
    class MyNeuron(bp.dyn.CondNeuGroupLTC):
        def __init__(self, size):
            super().__init__(size)
            self.IKv3 = Kv3(size)
            self.IIM  = IM(size)
"""

import brainpy as bp
import brainpy.math as bm


# ---------------------------------------------------------------------------
# Helper: safe exponential to avoid overflow
# ---------------------------------------------------------------------------

def _safe_exp(x):
    """Exponential clipped to avoid overflow."""
    return bm.exp(bm.clip(x, -50.0, 50.0))


def _sigmoid(v, v_half, k):
    """Boltzmann sigmoid: 1 / (1 + exp(-(v - v_half) / k))."""
    return 1.0 / (1.0 + _safe_exp(-(v - v_half) / k))


# ===========================================================================
# Kv3 -- Fast Delayed Rectifier K+ Channel
# ===========================================================================

class Kv3(bp.dyn.IonChannel):
    """
    Kv3 (Shaw-related) fast delayed rectifier potassium channel.

    Critical for PV+ fast-spiking interneurons: enables rapid repolarisation
    and high-frequency firing.

    I_Kv3 = g_Kv3 * n**2 * (V - E_K)
    """
    master_type = (bp.dyn.CondNeuGroup, bp.dyn.CondNeuGroupLTC)

    def __init__(self, size, g_max=1e-3, E=-85.0, method='exp_auto', **kwargs):
        super().__init__(size, **kwargs)
        self.g_max = g_max
        self.E = E
        self.n = bm.Variable(bm.zeros(self.num))
        self.integral = bp.odeint(self._dn, method=method)

    def _dn(self, n, t, V):
        n_inf = _sigmoid(V, -12.4, 6.8)
        tau_n = 0.25 + 4.35 * _safe_exp(-bm.abs(V + 16.0) / 40.0)
        return (n_inf - n) / bm.maximum(tau_n, 0.01)

    def reset_state(self, V, batch_size=None):
        n_inf = _sigmoid(V, -12.4, 6.8)
        self.n.value = n_inf if batch_size is None else bm.broadcast_to(n_inf, (batch_size,) + self.n.shape)

    def update(self, V):
        self.n.value = self.integral(self.n.value, bp.share['t'], V, bp.share['dt'])

    def current(self, V):
        return self.g_max * self.n ** 2 * (V - self.E)


# ===========================================================================
# IM -- Muscarinic (M-type) K+ Channel
# ===========================================================================

class IM(bp.dyn.IonChannel):
    """
    Muscarinic (M-type) potassium channel (Kv7 / KCNQ).
    I_M = g_M * p * (V - E_K)
    """
    master_type = (bp.dyn.CondNeuGroup, bp.dyn.CondNeuGroupLTC)

    def __init__(self, size, g_max=1e-5, E=-80.0, method='exp_auto', **kwargs):
        super().__init__(size, **kwargs)
        self.g_max = g_max
        self.E = E
        self.p = bm.Variable(bm.zeros(self.num))
        self.integral = bp.odeint(self._dp, method=method)

    def _dp(self, p, t, V):
        p_inf = _sigmoid(V, -35.0, 10.0)
        tau_p = 1000.0 / (3.3 * (_safe_exp((V + 35.0) / 20.0) +
                                  _safe_exp(-(V + 35.0) / 20.0)))
        tau_p = bm.clip(tau_p, 10.0, 1000.0)
        return (p_inf - p) / tau_p

    def reset_state(self, V, batch_size=None):
        p_inf = _sigmoid(V, -35.0, 10.0)
        self.p.value = p_inf if batch_size is None else bm.broadcast_to(p_inf, (batch_size,) + self.p.shape)

    def update(self, V):
        self.p.value = self.integral(self.p.value, bp.share['t'], V, bp.share['dt'])

    def current(self, V):
        return self.g_max * self.p * (V - self.E)


# ===========================================================================
# IAHP -- Ca2+-dependent K+ Channel (Afterhyperpolarization)
# ===========================================================================

class IAHP(bp.dyn.IonChannel):
    """
    Calcium-dependent potassium channel (medium AHP).
    Simplified: uses voltage as proxy for Ca2+ concentration.
    I_AHP = g_AHP * q * (V - E_K)
    """
    master_type = (bp.dyn.CondNeuGroup, bp.dyn.CondNeuGroupLTC)

    def __init__(self, size, g_max=1e-5, E=-80.0, method='exp_auto', **kwargs):
        super().__init__(size, **kwargs)
        self.g_max = g_max
        self.E = E
        self.q = bm.Variable(bm.zeros(self.num))
        self.integral = bp.odeint(self._dq, method=method)

    def _dq(self, q, t, V):
        q_inf = _sigmoid(V, -20.0, 5.0)
        tau_q = 100.0 + 500.0 / (1.0 + _safe_exp((V + 40.0) / 5.0))
        return (q_inf - q) / tau_q

    def reset_state(self, V, batch_size=None):
        q_inf = _sigmoid(V, -20.0, 5.0)
        self.q.value = q_inf if batch_size is None else bm.broadcast_to(q_inf, (batch_size,) + self.q.shape)

    def update(self, V):
        self.q.value = self.integral(self.q.value, bp.share['t'], V, bp.share['dt'])

    def current(self, V):
        return self.g_max * self.q * (V - self.E)


# ===========================================================================
# IT -- T-type Ca2+ Channel (Low-Voltage-Activated)
# ===========================================================================

class IT(bp.dyn.IonChannel):
    """
    T-type (low-voltage-activated) calcium channel.
    I_T = g_T * m**2 * h * (V - E_Ca)
    """
    master_type = (bp.dyn.CondNeuGroup, bp.dyn.CondNeuGroupLTC)

    def __init__(self, size, g_max=1e-4, E=120.0, method='exp_auto', **kwargs):
        super().__init__(size, **kwargs)
        self.g_max = g_max
        self.E = E
        self.m = bm.Variable(bm.zeros(self.num))
        self.h = bm.Variable(bm.ones(self.num))
        self.int_m = bp.odeint(self._dm, method=method)
        self.int_h = bp.odeint(self._dh, method=method)

    def _dm(self, m, t, V):
        m_inf = _sigmoid(V, -57.0, 6.2)
        tau_m = 0.13 + 3.68 / (1.0 + _safe_exp((V + 27.0) / (-10.0))) + \
                3.68 / (1.0 + _safe_exp(-(V + 102.0) / 15.0))
        tau_m = bm.maximum(tau_m, 0.1)
        return (m_inf - m) / tau_m

    def _dh(self, h, t, V):
        h_inf = _sigmoid(V, -81.0, -4.0)
        tau_h = 8.2 + (56.6 + 0.27 * _safe_exp((V + 48.0) / 4.0)) / \
                (1.0 + _safe_exp((V + 22.0) / (-5.0)))
        tau_h = bm.maximum(tau_h, 0.5)
        return (h_inf - h) / tau_h

    def reset_state(self, V, batch_size=None):
        m_inf = _sigmoid(V, -57.0, 6.2)
        h_inf = _sigmoid(V, -81.0, -4.0)
        if batch_size is None:
            self.m.value = m_inf
            self.h.value = h_inf
        else:
            self.m.value = bm.broadcast_to(m_inf, (batch_size,) + self.m.shape)
            self.h.value = bm.broadcast_to(h_inf, (batch_size,) + self.h.shape)

    def update(self, V):
        t, dt = bp.share['t'], bp.share['dt']
        self.m.value = self.int_m(self.m.value, t, V, dt)
        self.h.value = self.int_h(self.h.value, t, V, dt)

    def current(self, V):
        return self.g_max * self.m ** 2 * self.h * (V - self.E)


# ===========================================================================
# ICaL -- L-type Ca2+ Channel (High-Voltage-Activated)
# ===========================================================================

class ICaL(bp.dyn.IonChannel):
    """
    L-type (high-voltage-activated) calcium channel.
    I_CaL = g_CaL * m**2 * (V - E_Ca)
    """
    master_type = (bp.dyn.CondNeuGroup, bp.dyn.CondNeuGroupLTC)

    def __init__(self, size, g_max=1e-4, E=120.0, method='exp_auto', **kwargs):
        super().__init__(size, **kwargs)
        self.g_max = g_max
        self.E = E
        self.m = bm.Variable(bm.zeros(self.num))
        self.integral = bp.odeint(self._dm, method=method)

    def _dm(self, m, t, V):
        m_inf = _sigmoid(V, -10.0, 6.5)
        tau_m = 1.5 + 10.0 / (1.0 + _safe_exp((V + 25.0) / 10.0))
        tau_m = bm.maximum(tau_m, 0.5)
        return (m_inf - m) / tau_m

    def reset_state(self, V, batch_size=None):
        m_inf = _sigmoid(V, -10.0, 6.5)
        self.m.value = m_inf if batch_size is None else bm.broadcast_to(m_inf, (batch_size,) + self.m.shape)

    def update(self, V):
        self.m.value = self.integral(self.m.value, bp.share['t'], V, bp.share['dt'])

    def current(self, V):
        return self.g_max * self.m ** 2 * (V - self.E)


# ===========================================================================
# IH -- Hyperpolarization-Activated Cation Channel (HCN)
# ===========================================================================

class IH(bp.dyn.IonChannel):
    """
    Hyperpolarization-activated cation current (I_h / HCN).
    I_H = g_H * r * (V - E_H)  where E_H ~ -40 mV (mixed cation)
    """
    master_type = (bp.dyn.CondNeuGroup, bp.dyn.CondNeuGroupLTC)

    def __init__(self, size, g_max=1e-5, E=-40.0, method='exp_auto', **kwargs):
        super().__init__(size, **kwargs)
        self.g_max = g_max
        self.E = E
        self.r = bm.Variable(bm.zeros(self.num))
        self.integral = bp.odeint(self._dr, method=method)

    def _dr(self, r, t, V):
        r_inf = _sigmoid(V, -80.0, -10.0)
        tau_r = 100.0 + 1000.0 / (_safe_exp((V + 70.0) / 20.0) +
                                    _safe_exp(-(V + 70.0) / 20.0))
        tau_r = bm.clip(tau_r, 10.0, 5000.0)
        return (r_inf - r) / tau_r

    def reset_state(self, V, batch_size=None):
        r_inf = _sigmoid(V, -80.0, -10.0)
        self.r.value = r_inf if batch_size is None else bm.broadcast_to(r_inf, (batch_size,) + self.r.shape)

    def update(self, V):
        self.r.value = self.integral(self.r.value, bp.share['t'], V, bp.share['dt'])

    def current(self, V):
        return self.g_max * self.r * (V - self.E)


# ===========================================================================
# Channel Registry -- for agent to look up available channels
# ===========================================================================

CHANNEL_REGISTRY = {
    "Kv3": {
        "class": Kv3,
        "ion": "K",
        "description": "Fast delayed rectifier (Kv3/Shaw). Enables high-frequency "
                       "firing in PV+ FS interneurons via rapid repolarisation.",
        "key_param": "g_max",
        "typical_range": (1e-4, 0.1),
        "neuron_types": ["PV+ fast-spiking", "chandelier"],
    },
    "IM": {
        "class": IM,
        "ion": "K",
        "description": "Muscarinic M-type (KCNQ/Kv7). Slow non-inactivating K+ "
                       "current producing spike-frequency adaptation.",
        "key_param": "g_max",
        "typical_range": (1e-6, 1e-3),
        "neuron_types": ["L5 pyramidal", "regular-spiking"],
    },
    "IAHP": {
        "class": IAHP,
        "ion": "K",
        "description": "Ca2+-dependent K+ (medium AHP). Mediates medium "
                       "afterhyperpolarization and adaptation.",
        "key_param": "g_max",
        "typical_range": (1e-6, 1e-3),
        "neuron_types": ["L5 pyramidal", "regular-spiking"],
    },
    "IT": {
        "class": IT,
        "ion": "Ca",
        "description": "T-type Ca2+ (low-voltage-activated). Produces rebound "
                       "bursting in SST+ LTS interneurons.",
        "key_param": "g_max",
        "typical_range": (1e-5, 1e-2),
        "neuron_types": ["SST+ low-threshold-spiking", "thalamic relay"],
    },
    "ICaL": {
        "class": ICaL,
        "ion": "Ca",
        "description": "L-type Ca2+ (high-voltage-activated). Sustained calcium "
                       "entry during depolarisation.",
        "key_param": "g_max",
        "typical_range": (1e-5, 1e-2),
        "neuron_types": ["L5 pyramidal", "thalamic"],
    },
    "IH": {
        "class": IH,
        "ion": "mixed",
        "description": "HCN (I_h). Hyperpolarisation-activated cation current "
                       "producing sag and rebound depolarisation.",
        "key_param": "g_max",
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
                     f"(range: {info['typical_range'][0]:.0e}-{info['typical_range'][1]:.0e} S/cm2)")
        lines.append(f"    Neuron types: {', '.join(info['neuron_types'])}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(list_channels())

    # Quick smoke test: build a PV+ FS cell with Kv3
    # BrainPy equivalent of the original Jaxley smoke test:
    #   comp = jx.Compartment()
    #   comp.insert(Na()); comp.insert(K()); comp.insert(Leak())
    #   comp.insert(Kv3()); comp.insert(IM())
    import numpy as np

    class TestPVNeuron(bp.dyn.CondNeuGroupLTC):
        def __init__(self, size):
            super().__init__(size, V_initializer=bp.init.Constant(-65.))
            self.INa = bp.dyn.INa_HH1952(size, E=50.0, g_max=0.12)
            self.IK = bp.dyn.IK_HH1952(size, E=-77.0, g_max=0.036)
            self.IL = bp.dyn.IL(size, E=-54.387, g_max=0.0003)
            self.IKv3 = Kv3(size, g_max=1e-3)
            self.IIM = IM(size, g_max=1e-5)

    model = TestPVNeuron(1)
    runner = bp.DSRunner(model, monitors=['V'])
    inputs = np.ones(int(100.0 / bm.dt)) * 6.0  # 100 ms, 6 nA
    runner.run(inputs=inputs)

    print(f"\nSmoke test passed!")
    print(f"  V range: [{float(runner.mon['V'].min()):.1f}, {float(runner.mon['V'].max()):.1f}] mV")
    print(f"  Timesteps: {len(runner.mon['ts'])}")