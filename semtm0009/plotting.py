"""Plotting helpers with consistent stability semantics."""

from __future__ import annotations

from collections.abc import Iterable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from .analysis import Equilibrium, simulate
from .continuation import ContinuationBranch
from .types import FloatArray, ParameterSet, RHS


def state_grid(
    x_limits: tuple[float, float],
    y_limits: tuple[float, float],
    *,
    nx: int = 30,
    ny: int = 30,
) -> tuple[FloatArray, FloatArray]:
    """Create a rectangular sampling grid in a two-dimensional phase space."""
    x_values = np.linspace(*x_limits, nx)
    y_values = np.linspace(*y_limits, ny)
    xx, yy = np.meshgrid(x_values, y_values)
    return np.asarray(xx, dtype=float), np.asarray(yy, dtype=float)


def evaluate_vector_field(
    rhs: RHS[ParameterSet],
    parameters: ParameterSet,
    xx: FloatArray,
    yy: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    """Evaluate both components of a two-state ODE on a state grid."""
    if xx.shape != yy.shape:
        raise ValueError("xx and yy must have the same shape")
    coordinates = np.column_stack([xx.ravel(), yy.ravel()])
    velocity = np.asarray([rhs(0.0, state, parameters) for state in coordinates])
    if velocity.shape != (coordinates.shape[0], 2):
        raise ValueError("phase-space helpers require a two-state ODE")
    return velocity[:, 0].reshape(xx.shape), velocity[:, 1].reshape(yy.shape)


def plot_vector_field(
    xx: FloatArray,
    yy: FloatArray,
    dx: FloatArray,
    dy: FloatArray,
    *,
    ax: Axes,
    mode: str = "stream",
) -> Axes:
    """Plot state-space directions as streamlines or normalized arrows."""
    if mode == "stream":
        speed = np.hypot(dx, dy)
        ax.streamplot(xx, yy, dx, dy, color=np.log1p(speed), cmap="Greys", density=1.0)
    elif mode == "quiver":
        speed = np.hypot(dx, dy)
        scale = np.where(speed > 0, speed, 1.0)
        ax.quiver(xx, yy, dx / scale, dy / scale, color="#554987", alpha=0.75)
    else:
        raise ValueError("mode must be 'stream' or 'quiver'")
    return ax


def plot_nullclines(
    xx: FloatArray,
    yy: FloatArray,
    dx: FloatArray,
    dy: FloatArray,
    *,
    ax: Axes,
) -> Axes:
    """Plot the zero contours of each derivative component."""
    ax.contour(xx, yy, dx, levels=[0], colors=["#995CD0"], linewidths=2)
    ax.contour(xx, yy, dy, levels=[0], colors=["#4DAF4A"], linewidths=2)
    return ax


def plot_trajectories(
    rhs: RHS[ParameterSet],
    parameters: ParameterSet,
    initial_states: Iterable[Iterable[float]],
    t_span: tuple[float, float],
    *,
    ax: Axes,
    samples: int = 600,
) -> Axes:
    """Integrate and plot trajectories in phase space."""
    times = np.linspace(*t_span, samples)
    for initial_state in initial_states:
        solution = simulate(rhs, initial_state, t_span, parameters, t_eval=times)
        ax.plot(solution.y[0], solution.y[1], color="#201D30", linewidth=1.4)
    return ax


def phase_portrait(
    rhs: RHS[ParameterSet],
    parameters: ParameterSet,
    x_limits: tuple[float, float],
    y_limits: tuple[float, float],
    *,
    ax: Axes | None = None,
    density: int = 30,
) -> Axes:
    """Draw normalized flow and both zero-derivative nullclines."""
    if ax is None:
        _, ax = plt.subplots()
    xx, yy = state_grid(x_limits, y_limits, nx=density, ny=density)
    dx, dy = evaluate_vector_field(rhs, parameters, xx, yy)
    plot_vector_field(xx, yy, dx, dy, ax=ax)
    plot_nullclines(xx, yy, dx, dy, ax=ax)
    ax.set_xlim(x_limits)
    ax.set_ylim(y_limits)
    return ax


def plot_equilibria(
    equilibria: Iterable[Equilibrium],
    *,
    ax: Axes,
    size: float = 80,
    stable_color: str = "#69B64B",
) -> Axes:
    """Plot equilibria with a marker that communicates local classification."""
    legend_seen: set[str] = set()
    for equilibrium in equilibria:
        classification = equilibrium.classification
        if classification.startswith("stable"):
            marker, face, label = "o", stable_color, "stable"
        elif classification == "saddle":
            marker, face, label = "X", "#EC7744", "saddle"
        elif classification.startswith("unstable"):
            marker, face, label = "o", "none", "unstable"
        else:
            marker, face, label = "D", "none", "non-hyperbolic"
        x_coordinate, y_coordinate = equilibrium.state
        ax.scatter(
            x_coordinate,
            y_coordinate,
            marker=marker,
            s=size,
            facecolors=face,
            edgecolors="#201D30",
            linewidths=1.2,
            zorder=8,
            label=label if label not in legend_seen else None,
        )
        legend_seen.add(label)
    return ax


def equilibrium_branch(
    branch: ContinuationBranch,
    state_index: int = 0,
    *,
    ax: Axes | None = None,
) -> Axes:
    """Plot stable branches solid and unstable branches dashed."""
    if ax is None:
        _, ax = plt.subplots()
    for stability, style, label in [(True, "-", "stable"), (False, "--", "unstable")]:
        mask = (branch.stable == stability) & branch.converged
        ax.plot(
            branch.parameters[mask],
            branch.states[mask, state_index],
            style,
            color="#554987",
            linewidth=2,
            label=label,
        )
    ax.set_xlabel(branch.parameter_name)
    ax.set_ylabel(f"equilibrium state {state_index + 1}")
    return ax
