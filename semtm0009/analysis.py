"""Simulation, equilibrium finding, and local stability analysis."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol, cast

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root

from .types import ComplexArray, FloatArray, ParameterSet, RHS


class SimulationResult(Protocol):
    """The checked subset of a SciPy ODE result used by the workshops."""

    t: FloatArray
    y: FloatArray
    success: bool
    message: str


@dataclass(frozen=True)
class Equilibrium:
    """A numerically verified equilibrium and its local stability evidence."""

    state: FloatArray
    residual_norm: float
    eigenvalues: ComplexArray
    classification: str
    stable: bool


def simulate(
    rhs: RHS[ParameterSet],
    initial_state: Iterable[float],
    t_span: tuple[float, float],
    parameters: ParameterSet,
    *,
    t_eval: FloatArray | None = None,
    rtol: float = 1e-8,
    atol: float = 1e-10,
    method: str = "LSODA",
) -> SimulationResult:
    """Integrate an ODE and fail loudly if the solver does not converge."""
    solution = solve_ivp(
        lambda t, y: rhs(t, y, parameters),
        t_span,
        np.asarray(initial_state, dtype=float),
        t_eval=t_eval,
        rtol=rtol,
        atol=atol,
        method=method,
    )
    if not solution.success:
        raise RuntimeError(f"ODE integration failed: {solution.message}")
    return cast(SimulationResult, solution)


def numerical_jacobian(
    rhs: RHS[ParameterSet],
    state: Iterable[float],
    parameters: ParameterSet,
    *,
    relative_step: float = 1e-6,
) -> FloatArray:
    """Central-difference Jacobian of the vector field with respect to state."""
    point = np.asarray(state, dtype=float)
    jacobian = np.empty((point.size, point.size), dtype=float)
    for column in range(point.size):
        step = relative_step * max(1.0, abs(point[column]))
        offset = np.zeros_like(point)
        offset[column] = step
        jacobian[:, column] = (
            rhs(0.0, point + offset, parameters)
            - rhs(0.0, point - offset, parameters)
        ) / (2.0 * step)
    return jacobian


def classify_eigenvalues(eigenvalues: Iterable[complex], tolerance: float = 1e-7) -> str:
    """Classify a hyperbolic equilibrium from Jacobian eigenvalues."""
    values = np.asarray(eigenvalues, dtype=complex)
    real = values.real
    if np.any(np.abs(real) <= tolerance):
        return "non-hyperbolic"
    if np.any(real > 0) and np.any(real < 0):
        return "saddle"
    stable = bool(np.all(real < 0))
    oscillatory = bool(np.any(np.abs(values.imag) > tolerance))
    if stable and oscillatory:
        return "stable spiral"
    if stable:
        return "stable node"
    if oscillatory:
        return "unstable spiral"
    return "unstable node"


def _make_equilibrium(
    rhs: RHS[ParameterSet], state: FloatArray, parameters: ParameterSet, tolerance: float
) -> Equilibrium:
    residual = float(np.linalg.norm(rhs(0.0, state, parameters), ord=np.inf))
    jacobian = numerical_jacobian(rhs, state, parameters)
    eigenvalues = np.asarray(np.linalg.eigvals(jacobian), dtype=np.complex128)
    classification = classify_eigenvalues(eigenvalues, tolerance=tolerance)
    stable = bool(np.all(eigenvalues.real < -tolerance))
    return Equilibrium(state, residual, eigenvalues, classification, stable)


def find_equilibria(
    rhs: RHS[ParameterSet],
    guesses: Iterable[Iterable[float]],
    parameters: ParameterSet,
    *,
    residual_tolerance: float = 1e-8,
    duplicate_tolerance: float = 1e-5,
) -> list[Equilibrium]:
    """Find and deduplicate equilibria from a collection of initial guesses."""
    equilibria: list[Equilibrium] = []
    for guess in guesses:
        result = root(lambda state: rhs(0.0, state, parameters), np.asarray(guess))
        if not result.success:
            continue
        state = np.asarray(result.x, dtype=float)
        residual = np.linalg.norm(rhs(0.0, state, parameters), ord=np.inf)
        if residual > residual_tolerance:
            continue
        if any(np.linalg.norm(state - item.state) < duplicate_tolerance for item in equilibria):
            continue
        equilibria.append(
            _make_equilibrium(rhs, state, parameters, residual_tolerance)
        )
    equilibria.sort(key=lambda item: tuple(item.state))
    return equilibria
