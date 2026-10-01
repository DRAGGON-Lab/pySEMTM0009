"""Two-state persistent-sodium plus potassium neuron used in Practical 1.

The dynamic state is ordered ``[V, n]``: membrane potential in mV followed by
the dimensionless potassium activation gate.  Current and conductances are
expressed per unit membrane area, giving time in ms with ``C=1``.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from .types import FloatArray


IZHIKEVICH_PRACTICAL_PARAMETERS: dict[str, float] = {
    "C": 1.0,          # uF / cm^2
    "g_L": 8.0,       # mS / cm^2
    "g_Na": 20.0,     # mS / cm^2
    "g_K": 10.0,      # mS / cm^2
    "E_L": -80.0,     # mV
    "E_Na": 60.0,     # mV
    "E_K": -90.0,     # mV
    "V_m": -20.0,     # mV
    "k_m": 15.0,      # mV
    "V_n": -25.0,     # mV
    "k_n": 5.0,       # mV
    "tau_n": 1.0,     # ms
    "I": 0.0,         # uA / cm^2
}


def _parameter(parameters: Mapping[str, float], name: str) -> float:
    try:
        return float(parameters[name])
    except KeyError as error:
        raise KeyError(f"reduced-neuron parameter {name!r} is required") from error


def steady_gate(
    voltage: float | FloatArray, half_voltage: float, slope: float
) -> float | FloatArray:
    """Boltzmann steady-state activation, evaluated without avoidable overflow."""
    argument = np.clip((half_voltage - np.asarray(voltage)) / slope, -700.0, 700.0)
    return 1.0 / (1.0 + np.exp(argument))


def m_infinity(
    voltage: float | FloatArray, parameters: Mapping[str, float]
) -> float | FloatArray:
    return steady_gate(voltage, _parameter(parameters, "V_m"), _parameter(parameters, "k_m"))


def n_infinity(
    voltage: float | FloatArray, parameters: Mapping[str, float]
) -> float | FloatArray:
    return steady_gate(voltage, _parameter(parameters, "V_n"), _parameter(parameters, "k_n"))


def reduced_neuron(
    _time: float, state: FloatArray, parameters: Mapping[str, float]
) -> FloatArray:
    """Return ``[dV/dt, dn/dt]`` for a plain ``[V, n]`` state vector."""
    values = np.asarray(state, dtype=float)
    if values.shape != (2,):
        raise ValueError("reduced_neuron state must be a length-2 vector ordered [V, n]")
    voltage, gate = values
    current = _parameter(parameters, "I")
    leak = _parameter(parameters, "g_L") * (voltage - _parameter(parameters, "E_L"))
    sodium = (
        _parameter(parameters, "g_Na")
        * m_infinity(voltage, parameters)
        * (voltage - _parameter(parameters, "E_Na"))
    )
    potassium = (
        _parameter(parameters, "g_K") * gate * (voltage - _parameter(parameters, "E_K"))
    )
    d_voltage = (current - leak - sodium - potassium) / _parameter(parameters, "C")
    d_gate = (n_infinity(voltage, parameters) - gate) / _parameter(parameters, "tau_n")
    return np.asarray([d_voltage, d_gate], dtype=float)


def voltage_nullcline(
    voltage: float | FloatArray, parameters: Mapping[str, float]
) -> float | FloatArray:
    """Analytical ``dV/dt=0`` nullcline, returned as ``n(V)``."""
    voltage = np.asarray(voltage, dtype=float)
    numerator = (
        _parameter(parameters, "I")
        - _parameter(parameters, "g_L") * (voltage - _parameter(parameters, "E_L"))
        - _parameter(parameters, "g_Na")
        * m_infinity(voltage, parameters)
        * (voltage - _parameter(parameters, "E_Na"))
    )
    denominator = _parameter(parameters, "g_K") * (voltage - _parameter(parameters, "E_K"))
    return numerator / denominator


def analytical_jacobian(state: FloatArray, parameters: Mapping[str, float]) -> FloatArray:
    """Exact state Jacobian of :func:`reduced_neuron`."""
    voltage, gate = np.asarray(state, dtype=float)
    capacitance = _parameter(parameters, "C")
    tau = _parameter(parameters, "tau_n")
    m_value = float(m_infinity(voltage, parameters))
    n_value = float(n_infinity(voltage, parameters))
    dm = m_value * (1.0 - m_value) / _parameter(parameters, "k_m")
    dn = n_value * (1.0 - n_value) / _parameter(parameters, "k_n")
    d_v_d_v = -(
        _parameter(parameters, "g_L")
        + _parameter(parameters, "g_Na")
        * (dm * (voltage - _parameter(parameters, "E_Na")) + m_value)
        + _parameter(parameters, "g_K") * gate
    ) / capacitance
    d_v_d_n = -_parameter(parameters, "g_K") * (
        voltage - _parameter(parameters, "E_K")
    ) / capacitance
    return np.asarray([[d_v_d_v, d_v_d_n], [dn / tau, -1.0 / tau]])
