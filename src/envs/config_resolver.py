"""Locate the MATE checkout and its scenario files.

``src/`` holds the world model only.  The simulator, the scenario YAMLs, the
wrappers and the builtin target agents all come from ``MATE-main`` -- whether
that sits inside the repository or beside it.
"""

import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_PARENT = os.path.dirname(_REPO)

# Searched in order; the first directory that actually holds a ``mate`` package wins.
_CANDIDATE_ROOTS = (
    os.path.join(_REPO, 'MATE-main'),
    os.path.join(_PARENT, 'MATE-main'),
    os.path.join(_PARENT, 'MATE', 'MATE-main'),
)


def mate_root():
    """Directory holding the ``mate`` package.  ``MATE_ROOT`` overrides the search."""

    override = os.environ.get('MATE_ROOT')
    if override:
        return os.path.normpath(override)

    for candidate in _CANDIDATE_ROOTS:
        if os.path.isdir(os.path.join(candidate, 'mate')):
            return os.path.normpath(candidate)
    return os.path.normpath(_CANDIDATE_ROOTS[0])


def ensure_mate_importable():
    """Put the MATE checkout first on ``sys.path`` and return its root.

    Prepending matters: other MATE copies may be installed in the same
    interpreter (the older gym-based fork is pip-installed editable in the
    ``dima`` env), and the import has to resolve to this checkout.
    """

    root = mate_root()
    if not os.path.isdir(os.path.join(root, 'mate')):
        raise FileNotFoundError(
            f'No MATE checkout at {root!r}. Point MATE_ROOT at the MATE-main directory.'
        )
    while root in sys.path:
        sys.path.remove(root)
    sys.path.insert(0, root)
    return root


def assets_dir():
    """MATE's own scenario directory."""

    return os.path.join(ensure_mate_importable(), 'mate', 'assets')


def available_scenarios():
    """Scenario names shipped by the MATE checkout, e.g. ``MATE-4v8-9``."""

    return sorted(
        os.path.splitext(f)[0]
        for f in os.listdir(assets_dir())
        if f.endswith(('.yaml', '.json'))
    )


def resolve_scenario(name):
    """Normalize a scenario reference into something ``mate.make_environment`` accepts.

    Accepts a bare name (``MATE-4v8-9``), a file name (``MATE-4v8-9.yaml``), a
    shorthand (``4v8-9``, ``4C8T-9``) or an absolute path to a config file.
    """

    if os.path.isabs(name) and os.path.isfile(name):
        return name

    candidate = name if name.endswith(('.yaml', '.json')) else name + '.yaml'
    directory = assets_dir()

    if os.path.isfile(os.path.join(directory, candidate)):
        return candidate

    stem = os.path.splitext(candidate)[0].upper().replace('C', '').replace('T', '')
    for scenario in available_scenarios():
        if scenario.upper() == f'MATE-{stem}':
            return scenario + '.yaml'

    raise FileNotFoundError(
        f'Scenario {name!r} not found in {directory}. Available: {available_scenarios()}'
    )
