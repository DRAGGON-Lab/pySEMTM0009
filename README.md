# pySEMTM0009

Small, transparent Python utilities for SEMTM0009 phase-plane analysis,
simulation, equilibrium stability, and numerical continuation. The installed
distribution is named `pysemtm0009`; the import package remains `semtm0009`.

## Google Colab

For quick testing against the moving default branch:

```python
!pip install -q "git+https://github.com/DRAGGON-Lab/pySEMTM0009.git"
```

P1 temporarily follows `main` while the new figure-output API is under
pre-release testing. Pin the next reviewed full commit as soon as it is pushed;
do not use the earlier `3faa173...` revision because it predates that API.

Once a reviewed `v0.1.0` tag and PyPI release actually exist, the preferred
commands will be:

```python
!pip install -q "git+https://github.com/DRAGGON-Lab/pySEMTM0009.git@v0.1.0"
!pip install -q "pysemtm0009==0.1.0"
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

Install before importing `semtm0009`. A tag or full commit hash prevents
equations or numerical behavior changing silently between teaching sessions.

## Local development and build

The supported interpreters are Python 3.12 and 3.13. In the standalone package
repository, create an environment and run the same checks as CI:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -e ".[dev]" build twine
.venv/bin/python -m pytest
.venv/bin/python -m mypy
.venv/bin/python -m build
.venv/bin/python -m twine check dist/*
```

Runtime dependencies are NumPy, SciPy, and Matplotlib. The current continuation
backend is included and does not require pyoomph.

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

## Notebook figure output

Use the public renderer factory to keep notebook setup cells short while still
supporting inline teaching views and slide-ready files:

```python
from pathlib import Path
from semtm0009 import configure_figure_output

render_figure = configure_figure_output(
    "both",  # "inline", "files", or "both"
    Path.cwd() / "figures" / "generated",
)

# After constructing a Matplotlib figure:
render_figure(fig, "phase-portrait")
```

The callable validates its configuration, creates the output directory when
needed, writes PDF and PNG files at 180 dpi, displays an in-memory PNG in
IPython for `inline`/`both`, and closes the figure. Named DRAGGON palette values
are available as `d_purple`, `d_orange`, `d_green`, and `d_red`.

## Synchronizing from the workshop

The workshop `python/` directory is the upstream mirror. In the standalone
repository, preview and then apply a synchronization with:

```bash
./scripts/sync_from_workshop.sh /path/to/semtm0009-python-workshops/python
./scripts/sync_from_workshop.sh --apply /path/to/semtm0009-python-workshops/python
```

Repository-only `.github/`, `scripts/sync_from_workshop.sh`, and `graft/` are
protected. Git metadata, environments, caches, builds, and bytecode are
excluded. Review every dry run because synchronization uses deletion for
unprotected mirrored paths.

## Releasing with Trusted Publishing

The repository owner must configure a protected GitHub environment named
`pypi` and a PyPI Trusted Publisher for project `pysemtm0009`, owner
`DRAGGON-Lab`, repository `pySEMTM0009`, workflow `publish.yml`, and environment
`pypi`. Create `v0.1.0` only after CI is green on the reviewed release commit.
The workflow uses OpenID Connect and should not contain a long-lived PyPI token.
