"""Regression tests for the stable teaching API and compatibility shim."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import cast

import numpy as np
import semtm0009_continuation as legacy

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

from semtm0009 import (
    IZHIKEVICH_PRACTICAL_PARAMETERS,
    __version__,
    continue_equilibrium,
    continue_periodic_orbit,
    reduced_neuron,
)
from semtm0009.continuation import ContinuationBranch
from semtm0009.types import Parameters, RHS


def test_public_package_api_and_legacy_shim() -> None:
    assert __version__ == "0.1.0"
    assert callable(reduced_neuron)
    assert callable(continue_equilibrium)
    assert callable(continue_periodic_orbit)
    assert isinstance(IZHIKEVICH_PRACTICAL_PARAMETERS, dict)
    assert legacy.continue_equilibrium is continue_equilibrium
    assert legacy.continue_periodic_orbit is continue_periodic_orbit


def test_reduced_neuron_preserves_state_derivative_order() -> None:
    params = dict(IZHIKEVICH_PRACTICAL_PARAMETERS)
    rate = reduced_neuron(0.0, np.array([-65.0, 0.001]), params)
    assert rate.shape == (2,)
    assert np.all(np.isfinite(rate))
    np.testing.assert_allclose(
        rate,
        np.array([-1.6853170560830506, -0.0006646498695335219]),
        rtol=1e-12,
        atol=1e-14,
    )


def test_five_point_equilibrium_branch_has_consistent_arrays() -> None:
    params = dict(IZHIKEVICH_PRACTICAL_PARAMETERS)
    branch = continue_equilibrium(
        cast(RHS[Parameters], reduced_neuron),
        [-65.952951263, 2.77173342e-4],
        "I",
        params,
        initial_parameter_step=0.05,
        arclength_step=0.15,
        max_points=5,
        parameter_bounds=(-1.0, 6.0),
    )
    assert isinstance(branch, ContinuationBranch)
    assert branch.parameter_name == "I"
    assert isinstance(branch.parameters, np.ndarray)
    assert isinstance(branch.states, np.ndarray)
    assert isinstance(branch.eigenvalues, np.ndarray)
    assert isinstance(branch.stable, np.ndarray)
    assert isinstance(branch.converged, np.ndarray)
    assert branch.parameters.shape == (5,)
    assert branch.states.shape == (5, 2)
    assert branch.eigenvalues.shape == (5, 2)
    assert branch.stable.shape == (5,)
    assert branch.converged.shape == (5,)
    assert len(branch.labels) == 5
    assert np.issubdtype(branch.parameters.dtype, np.floating)
    assert np.issubdtype(branch.states.dtype, np.floating)
    assert np.issubdtype(branch.eigenvalues.dtype, np.complexfloating)
    assert np.issubdtype(branch.stable.dtype, np.bool_)
    assert np.issubdtype(branch.converged.dtype, np.bool_)
    assert np.all(branch.converged)
    assert np.all(np.isfinite(branch.parameters))
    assert np.all(np.isfinite(branch.states))
    assert np.all(np.isfinite(branch.eigenvalues))
