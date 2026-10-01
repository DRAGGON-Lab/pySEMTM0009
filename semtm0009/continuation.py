"""Small, transparent continuation routines for teaching and coursework."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root

from .analysis import numerical_jacobian
from .types import BoolArray, ComplexArray, FloatArray, Parameters, RHS


@dataclass(frozen=True)
class ContinuationBranch:
    parameter_name: str
    parameters: FloatArray
    states: FloatArray
    eigenvalues: ComplexArray
    stable: BoolArray
    labels: tuple[str, ...]
    converged: BoolArray


@dataclass(frozen=True)
class PeriodicBranch:
    parameter_name: str
    parameters: FloatArray
    periods: FloatArray
    minima: FloatArray
    maxima: FloatArray
    floquet_multipliers: ComplexArray
    stable: BoolArray
    converged: BoolArray


def _augmented_equilibrium(
    rhs: RHS[Parameters],
    state_parameter: FloatArray,
    parameter_name: str,
    base_parameters: Parameters,
    tangent: FloatArray,
    predictor: FloatArray,
) -> FloatArray:
    state = state_parameter[:-1]
    parameter_value = state_parameter[-1]
    parameters = dict(base_parameters)
    parameters[parameter_name] = parameter_value
    return np.asarray(
        np.r_[rhs(0.0, state, parameters), np.dot(state_parameter - predictor, tangent)],
        dtype=float,
    )


def _branch_point(
    rhs: RHS[Parameters],
    point: FloatArray,
    parameter_name: str,
    base_parameters: Parameters,
) -> tuple[ComplexArray, bool]:
    parameters = dict(base_parameters)
    parameters[parameter_name] = float(point[-1])
    eigenvalues = np.asarray(
        np.linalg.eigvals(numerical_jacobian(rhs, point[:-1], parameters)),
        dtype=np.complex128,
    )
    return eigenvalues, bool(np.all(eigenvalues.real < -1e-6))


def continue_equilibria(
    rhs: RHS[Parameters],
    initial_state: Iterable[float],
    parameter_name: str,
    parameters: Parameters,
    *,
    initial_parameter_step: float = 1e-2,
    arclength_step: float = 5e-2,
    max_points: int = 200,
    parameter_bounds: tuple[float, float] = (-np.inf, np.inf),
) -> ContinuationBranch:
    """Pseudo-arclength continuation of an equilibrium branch.

    The first two points are obtained by natural parameter continuation. Subsequent
    corrector steps solve the equilibrium equations plus a secant phase condition.
    """
    base = dict(parameters)
    first_parameter = float(base[parameter_name])
    first_solution = root(
        lambda state: rhs(0.0, state, base), np.asarray(initial_state, dtype=float)
    )
    if not first_solution.success:
        raise RuntimeError("Could not correct the initial equilibrium")
    first = np.r_[first_solution.x, first_parameter]

    second_parameters = dict(base)
    second_parameters[parameter_name] = first_parameter + initial_parameter_step
    second_solution = root(
        lambda state: rhs(0.0, state, second_parameters), first_solution.x
    )
    if not second_solution.success:
        raise RuntimeError("Could not construct the second continuation point")
    second = np.r_[second_solution.x, second_parameters[parameter_name]]

    points = [first, second]
    failed_parameter: float | None = None
    for _ in range(max_points - 2):
        tangent = points[-1] - points[-2]
        norm = np.linalg.norm(tangent)
        if norm == 0:
            break
        tangent /= norm
        predictor = points[-1] + arclength_step * tangent
        corrected = root(
            lambda value: _augmented_equilibrium(
                rhs, value, parameter_name, base, tangent, predictor
            ),
            predictor,
        )
        if not corrected.success or np.linalg.norm(corrected.fun, ord=np.inf) > 1e-7:
            failed_parameter = float(predictor[-1])
            break
        point = np.asarray(corrected.x)
        if not parameter_bounds[0] <= point[-1] <= parameter_bounds[1]:
            break
        points.append(point)

    array = np.asarray(points)
    eigens = []
    stability = []
    for point in array:
        eigenvalues, stable = _branch_point(rhs, point, parameter_name, base)
        eigens.append(eigenvalues)
        stability.append(stable)
    eigens_array = np.asarray(eigens)
    labels = ["" for _ in array]
    for index in range(1, len(array)):
        previous = eigens_array[index - 1]
        current = eigens_array[index]
        if np.min(np.abs(current.real)) < 5e-3:
            if np.any(np.abs(current.imag) > 1e-3):
                labels[index] = "candidate Hopf"
            else:
                labels[index] = "candidate fold"
        elif bool(stability[index]) != bool(stability[index - 1]):
            labels[index] = "stability change"
    converged_array = np.ones(len(array), dtype=bool)
    if failed_parameter is not None:
        array = np.vstack([array, np.r_[np.full(array.shape[1] - 1, np.nan), failed_parameter]])
        eigens_array = np.vstack(
            [eigens_array, np.full((1, eigens_array.shape[1]), np.nan + 0j)]
        )
        stability = [*stability, False]
        labels.append("non-converged")
        converged_array = np.r_[converged_array, False]
    return ContinuationBranch(
        parameter_name,
        array[:, -1],
        array[:, :-1],
        eigens_array,
        np.asarray(stability),
        tuple(labels),
        converged_array,
    )


def _shooting_residual(
    unknown: FloatArray,
    rhs: RHS[Parameters],
    parameter_name: str,
    parameter_value: float,
    base_parameters: Parameters,
    reference_state: FloatArray,
    reference_tangent: FloatArray,
) -> FloatArray:
    state = unknown[:-1]
    period = float(np.exp(unknown[-1]))
    parameters = dict(base_parameters)
    parameters[parameter_name] = parameter_value
    solution = solve_ivp(
        lambda t, y: rhs(t, y, parameters),
        (0.0, period),
        state,
        rtol=1e-8,
        atol=1e-10,
        method="LSODA",
    )
    closure = solution.y[:, -1] - state
    phase = np.dot(state - reference_state, reference_tangent)
    return np.asarray(np.r_[closure, phase], dtype=float)


def _floquet_multipliers(
    rhs: RHS[Parameters],
    state: FloatArray,
    period: float,
    parameters: Parameters,
) -> ComplexArray:
    dimension = state.size
    initial = np.r_[state, np.eye(dimension).ravel()]

    def variational(t: float, augmented: FloatArray) -> FloatArray:
        current_state = augmented[:dimension]
        matrix = augmented[dimension:].reshape(dimension, dimension)
        jacobian = numerical_jacobian(rhs, current_state, parameters)
        return np.asarray(
            np.r_[rhs(t, current_state, parameters), (jacobian @ matrix).ravel()],
            dtype=float,
        )

    solution = solve_ivp(
        variational, (0.0, period), initial, rtol=2e-7, atol=1e-9, method="LSODA"
    )
    monodromy = solution.y[dimension:, -1].reshape(dimension, dimension)
    return np.asarray(np.linalg.eigvals(monodromy), dtype=np.complex128)


def continue_periodic_orbits(
    rhs: RHS[Parameters],
    orbit_guess: FloatArray,
    time_guess: FloatArray,
    parameter_name: str,
    parameter_values: Iterable[float],
    parameters: Parameters,
) -> PeriodicBranch:
    """Continue periodic orbits by shooting with a phase condition.

    A converged time series supplies the first state and a tangent. Parameter values
    are traversed sequentially; failures are recorded rather than silently joined.
    """
    orbit = np.asarray(orbit_guess, dtype=float)
    times = np.asarray(time_guess, dtype=float)
    if orbit.ndim != 2 or orbit.shape[1] != times.size:
        raise ValueError("orbit_guess must have shape (n_states, n_times)")
    state = orbit[:, 0]
    period = float(times[-1] - times[0])
    tangent = orbit[:, 1] - orbit[:, 0]
    tangent /= np.linalg.norm(tangent)
    reference = state.copy()

    values = np.asarray(list(parameter_values), dtype=float)
    periods = np.full(values.shape, np.nan)
    minima = np.full((values.size, state.size), np.nan)
    maxima = np.full((values.size, state.size), np.nan)
    multipliers = np.full((values.size, state.size), np.nan + 0j)
    stable = np.zeros(values.size, dtype=bool)
    converged = np.zeros(values.size, dtype=bool)

    for index, value in enumerate(values):
        unknown = np.r_[state, np.log(period)]
        result = root(
            lambda candidate: _shooting_residual(
                candidate,
                rhs,
                parameter_name,
                float(value),
                parameters,
                reference,
                tangent,
            ),
            unknown,
        )
        if not result.success or np.linalg.norm(result.fun, ord=np.inf) > 1e-6:
            continue
        state = result.x[:-1]
        period = float(np.exp(result.x[-1]))
        current_parameters = dict(parameters)
        current_parameters[parameter_name] = float(value)
        sample_times = np.linspace(0.0, period, 300)
        solution = solve_ivp(
            lambda t, y: rhs(t, y, current_parameters),
            (0.0, period),
            state,
            t_eval=sample_times,
            rtol=1e-8,
            atol=1e-10,
            method="LSODA",
        )
        current_multipliers = _floquet_multipliers(
            rhs, state, period, current_parameters
        )
        nontrivial = np.delete(
            np.abs(current_multipliers), np.argmin(np.abs(current_multipliers - 1.0))
        )
        periods[index] = period
        minima[index] = solution.y.min(axis=1)
        maxima[index] = solution.y.max(axis=1)
        multipliers[index, : current_multipliers.size] = current_multipliers
        stable[index] = bool(np.all(nontrivial < 1.0 + 1e-4))
        converged[index] = True
        reference = state.copy()
        tangent = rhs(0.0, state, current_parameters)
        tangent /= np.linalg.norm(tangent)

    return PeriodicBranch(
        parameter_name, values, periods, minima, maxima, multipliers, stable, converged
    )
