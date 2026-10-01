"""Reference models used in the SEMTM0009 workshops."""

from __future__ import annotations

import numpy as np

from .types import (
    BrusselatorParameters,
    FHNParameters,
    FloatArray,
    LorenzParameters,
    MorrisLecarParameters,
    RepressilatorParameters,
    SEIRParameters,
    SIRParameters,
    ToggleParameters,
)


def fitzhugh_nagumo(
    _t: float, state: FloatArray, parameters: FHNParameters
) -> FloatArray:
    """FitzHugh--Nagumo dynamics with applied current ``I``."""
    v, w = state
    epsilon = parameters.get("epsilon", 0.08)
    current = parameters.get("I", 0.0)
    return np.array(
        [v - v**3 / 3.0 - w + current, epsilon * (v + 0.7 - 0.8 * w)]
    )


def toggle_switch(
    _t: float, state: FloatArray, parameters: ToggleParameters
) -> FloatArray:
    """Dimensionless mutually repressing genetic toggle switch."""
    u, v = state
    alpha_1 = parameters.get("alpha_1", 5.0)
    alpha_2 = parameters.get("alpha_2", 5.0)
    beta = parameters.get("beta", 4.0)
    gamma = parameters.get("gamma", 4.0)
    return np.array(
        [alpha_1 / (1.0 + v**beta) - u, alpha_2 / (1.0 + u**gamma) - v]
    )


def brusselator(
    _t: float, state: FloatArray, parameters: BrusselatorParameters
) -> FloatArray:
    """Dimensionless Brusselator; ``b`` is the continuation parameter."""
    x, y = state
    a = parameters.get("a", 1.0)
    b = parameters.get("b", 2.0)
    reaction = x * x * y
    return np.array([a - (b + 1.0) * x + reaction, b * x - reaction])


def morris_lecar_bursting(
    _t: float, state: FloatArray, parameters: MorrisLecarParameters
) -> FloatArray:
    """Morris--Lecar voltage/gate with a slow adaptation current.

    The first two equations are a compact conductance-based Morris--Lecar
    system.  The third, deliberately slow, state is the added mechanism that
    creates burst envelopes; base two-state Morris--Lecar models tonic
    excitability/spiking rather than genuine bursts.
    """
    voltage, gate, adaptation = state
    current = parameters.get("applied_current", 92.0)
    phi = parameters.get("phi", 0.04)
    slow_rate = parameters.get("slow_rate", 0.004)
    gain = parameters.get("adaptation_gain", 18.0)
    calcium = 4.4 * 0.5 * (1.0 + np.tanh((voltage + 1.2) / 18.0)) * (voltage - 120.0)
    potassium = 8.0 * gate * (voltage + 84.0)
    leak = 2.0 * (voltage + 60.0)
    d_voltage = current - calcium - potassium - leak - adaptation
    gate_inf = 0.5 * (1.0 + np.tanh((voltage - 2.0) / 30.0))
    gate_scale = np.cosh((voltage - 2.0) / 60.0)
    d_gate = phi * gate_scale * (gate_inf - gate)
    adaptation_inf = gain * 0.5 * (1.0 + np.tanh((voltage + 10.0) / 10.0))
    d_adaptation = slow_rate * (adaptation_inf - adaptation)
    return np.array([d_voltage, d_gate, d_adaptation])


def sir(_t: float, state: FloatArray, parameters: SIRParameters) -> FloatArray:
    """Closed-population susceptible--infectious--recovered model."""
    susceptible, infectious, recovered = state
    beta = parameters["beta"]
    gamma = parameters["gamma"]
    incidence = beta * susceptible * infectious
    return np.array([-incidence, incidence - gamma * infectious, gamma * infectious])


def seir(_t: float, state: FloatArray, parameters: SEIRParameters) -> FloatArray:
    """Closed-population susceptible--exposed--infectious--recovered model."""
    susceptible, exposed, infectious, recovered = state
    beta = parameters["beta"]
    sigma = parameters["sigma"]
    gamma = parameters["gamma"]
    incidence = beta * susceptible * infectious
    progression = sigma * exposed
    recovery = gamma * infectious
    return np.array([-incidence, incidence - progression, progression - recovery, recovery])


def lorenz(
    _t: float, state: FloatArray, parameters: LorenzParameters
) -> FloatArray:
    """Lorenz system in the conventional sigma--rho--beta parameterisation."""
    x, y, z = state
    sigma = parameters.get("sigma", 10.0)
    rho = parameters.get("rho", 28.0)
    beta = parameters.get("beta", 8.0 / 3.0)
    return np.array([sigma * (y - x), x * (rho - z) - y, x * y - beta * z])


def repressilator(
    _t: float, state: FloatArray, parameters: RepressilatorParameters
) -> FloatArray:
    """Reduced three-state ring oscillator with dimensionless first-order loss."""
    p_1, p_2, p_3 = state
    alpha = parameters.get("alpha", 10.0)
    hill = parameters.get("hill", 3.0)
    basal = parameters.get("basal", 0.0)
    def production(repressor: float) -> float:
        return float(basal + alpha / (1.0 + repressor**hill))
    return np.array(
        [production(p_3) - p_1, production(p_1) - p_2, production(p_2) - p_3]
    )
