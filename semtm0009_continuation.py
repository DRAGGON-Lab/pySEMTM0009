"""Compatibility shim for the pre-package continuation import.

New code should use ``from semtm0009 import continue_equilibrium``. This module
remains in the workshop checkout so existing notebooks do not break, but it is
not part of the standalone distribution built from ``python/``.
"""

from __future__ import annotations

from semtm0009.continuation_api import (
    ContinuationBranch,
    PeriodicBranch,
    continue_equilibrium,
    continue_periodic_orbit,
)


__all__ = [
    "ContinuationBranch",
    "PeriodicBranch",
    "continue_equilibrium",
    "continue_periodic_orbit",
]
