# pySEMTM0009

Small, transparent Python utilities for SEMTM0009 phase-plane analysis,
simulation, equilibrium stability, and numerical continuation. The installed
distribution is named `pysemtm0009`; the import package remains `semtm0009`.
The two names intentionally differ: use `pysemtm0009` with package installers
and `semtm0009` in Python code.

## Google Colab

Install the reviewed `v0.1.0` source directly from GitHub:

```python
%pip install -q "git+https://github.com/DRAGGON-Lab/pySEMTM0009.git@v0.1.0"
```

After the release is published to PyPI, the equivalent package install is:

```python
%pip install -q "pysemtm0009==0.1.0"
```

Then import the stable teaching API:

```python
from semtm0009 import (
    IZHIKEVICH_PRACTICAL_PARAMETERS,
    continue_equilibrium,
    continue_periodic_orbit,
    reduced_neuron,
)
```

No restart is normally required because this package does not replace Colab's
notebook runtime components. Install it before importing `semtm0009`.
Teaching notebooks must pin a release tag or full commit hash. Installing the
moving default branch can silently change equations or numerical behavior
between classes and makes results harder to reproduce.

## Local development and build

The supported interpreters are Python 3.12 and 3.13. Create an environment,
install the development extra, then run the same core checks as CI:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -e ".[dev]" build twine
.venv/bin/python -m pytest
.venv/bin/python -m mypy
.venv/bin/python -m build
.venv/bin/python -m twine check dist/*
```

Runtime dependencies are NumPy, SciPy, and Matplotlib. The continuation façade
is backend-independent; this release uses the included pseudo-arclength and
shooting routines and does not require pyoomph.

## Public continuation API

```python
branch = continue_equilibrium(
    reduced_neuron,
    initial_state=[-65.953, 2.77e-4],
    parameter_name="I",
    params={**IZHIKEVICH_PRACTICAL_PARAMETERS, "I": 0.0},
    initial_parameter_step=0.05,
    arclength_step=0.15,
    max_points=100,
    parameter_bounds=(-1.0, 6.0),
)
```

Returned branches contain ordinary NumPy arrays for parameters, states or
orbit envelopes, stability data, convergence flags, and bifurcation labels.

## Synchronizing from the workshop

The workshop's `python/` directory is the upstream mirror. Preview a sync,
review the itemized changes, and apply only after they look correct:

```bash
./scripts/sync_from_workshop.sh /path/to/semtm0009-python-workshops/python
./scripts/sync_from_workshop.sh --apply /path/to/semtm0009-python-workshops/python
```

The source argument may appear before or after `--apply`. Synchronization uses
archive mode and deletion, so all mirrored paths not explicitly protected are
workshop-owned. Repository-only `.github/`, `scripts/sync_from_workshop.sh`,
and `graft/` are protected from copying and deletion. Git metadata, virtual
environments, caches, build products, bytecode, and `.DS_Store` are excluded.

After changing production metadata, README content, ignore rules, or public
tests here, backport those files to the workshop before the next sync. The
scientific modules are copied unchanged.

## Releasing with Trusted Publishing

The repository owner must create a protected GitHub environment named `pypi`
(preferably with required reviewers), then configure a PyPI pending Trusted
Publisher for project `pysemtm0009`, owner `DRAGGON-Lab`, repository
`pySEMTM0009`, workflow `publish.yml`, and environment `pypi`.

Once CI is green on the reviewed release commit, create and push the matching
tag (for this version, `v0.1.0`). The publication workflow checks that the tag
matches the project version, rebuilds and validates the distributions, then
publishes through OpenID Connect. It uses no long-lived PyPI token. A pending
publisher does not reserve a project name; the name becomes claimed only after
the first successful publication.
