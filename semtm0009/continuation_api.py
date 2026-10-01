"""Stable public continuation façade for notebooks and external packages.

The functions return the NumPy-backed branch dataclasses from
``semtm0009.continuation``. A future backend can therefore be introduced behind
this module without changing student notebooks.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np

from .continuation import (
    ContinuationBranch,
    PeriodicBranch,
    continue_equilibria,
    continue_periodic_orbits,
)
from .types import FloatArray, Parameters, RHS


def continue_equilibrium(
    model: RHS[Parameters],
    initial_state: Iterable[float],
    parameter_name: str,
    params: Mapping[str, float],
    **options: Any,
) -> ContinuationBranch:
    """Continue equilibria through folds using the teaching backend."""
    if parameter_name not in params:
        raise KeyError(f"continuation parameter {parameter_name!r} is absent from params")
    return continue_equilibria(
        model, initial_state, parameter_name, dict(params), **options
    )


def continue_periodic_orbit(
    model: RHS[Parameters],
    orbit_guess: FloatArray,
    time_guess: FloatArray,
    parameter_name: str,
    params: Mapping[str, float],
    **options: Any,
) -> PeriodicBranch:
    """Continue a shooting orbit over explicit ``parameter_values``."""
    if parameter_name not in params:
        raise KeyError(f"continuation parameter {parameter_name!r} is absent from params")
    try:
        parameter_values = options.pop("parameter_values")
    except KeyError as error:
        raise TypeError("parameter_values is a required keyword option") from error
    if options:
        names = ", ".join(sorted(options))
        raise TypeError(f"unsupported periodic-continuation option(s): {names}")
    return continue_periodic_orbits(
        model,
        np.asarray(orbit_guess, dtype=float),
        np.asarray(time_guess, dtype=float),
        parameter_name,
        parameter_values,
        dict(params),
    )


__all__ = [
    "ContinuationBranch",
    "PeriodicBranch",
    "continue_equilibrium",
    "continue_periodic_orbit",
]

