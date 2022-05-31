import os
import pathlib
import itertools
import shutil

MONK_DATA_DIR = pathlib.Path(os.environ["MONK_DATA_DIR"]) / "monk"

if MONK_DATA_DIR == "":
    raise ValueError("MONK_DATA_DIR environment variable needs to be set")


def project_path(dir: str):
    assert dir != ""
    return MONK_DATA_DIR / dir

def project_view(dir: str):
    assert dir != ""
    return MONK_DATA_DIR / dir / "view"

def safe_clean_signac_project(dir: str, prepend_monk: bool = True):
    if prepend_monk:
        path = MONK_DATA_DIR / dir
    else:
        path = pathlib.Path(dir)
    signac_rc = path / "signac.rc"
    if not signac_rc.exists():
        raise FileNotFoundError(f"Cannot find {signac_rc}")
    else:
        print(f"You are about to delete {path}")
        print("This folder contains ...")
        idx = 0
        for child in path.iterdir():
            if idx >= 10:
                break
            print(child[-1])
            idx += 1
        print("Are you sure? y/N")
        if input() == "y":
            shutil.rmtree(path)


def grid(gridspec):
    """Yields the Cartesian product of a `dict` of iterables.

    The input ``gridspec`` is a dictionary whose keys correspond to
    parameter names. Each key is associated with an iterable of the
    values that parameter could take on. The result is a sequence of
    dictionaries where each dictionary has one of the unique combinations
    of the parameter values.
    """
    for values in itertools.product(*gridspec.values()):
        yield dict(zip(gridspec.keys(), values))