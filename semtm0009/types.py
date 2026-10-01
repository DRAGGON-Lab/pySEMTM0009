"""PEP 484 types that connect the mathematical and Python interfaces."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol, TypeAlias, TypeVar, TypedDict

import numpy as np
from numpy.typing import NDArray

FloatArray: TypeAlias = NDArray[np.float64]
ComplexArray: TypeAlias = NDArray[np.complex128]
BoolArray: TypeAlias = NDArray[np.bool_]
Parameters = Mapping[str, float]
ParameterSet = TypeVar("ParameterSet", contravariant=True)


class RHS(Protocol[ParameterSet]):
    """A typed representation of ``f(t, x; theta) -> dx/dt``."""

    def __call__(
        self, time: float, state: FloatArray, parameters: ParameterSet
    ) -> FloatArray: ...


class FHNParameters(TypedDict):
    I: float
    epsilon: float


class ToggleParameters(TypedDict):
    alpha_1: float
    alpha_2: float
    beta: float
    gamma: float


class SIRParameters(TypedDict):
    beta: float
    gamma: float


class SEIRParameters(SIRParameters):
    sigma: float


class LorenzParameters(TypedDict):
    sigma: float
    rho: float
    beta: float


class RepressilatorParameters(TypedDict):
    alpha: float
    hill: float
    basal: float


class BrusselatorParameters(TypedDict):
    a: float
    b: float


class MorrisLecarParameters(TypedDict):
    applied_current: float
    phi: float
    slow_rate: float
    adaptation_gain: float
