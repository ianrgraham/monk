"""Script to generate statepoints of the workspace"""

import os
import sys
import shutil
import pathlib

import click
import signac

from monk import workflow, grid


@click.command()
@click.option("--num",
              type=int,
              default=16384,
              show_default=True,
              help="Number of particles in the simulation")
@click.option("--pressure",
              type=float,
              default=1.0,
              show_default=True,
              help='Pressure to apply to simulations')
@click.option('--clear-workspace',
              is_flag=True,
              default=False,
              help='Clears all data from the workspace of the project')
def main(num: int, pressure: float, clear_workspace: bool):
    # grabs the project configuration
    config = workflow.get_config()

    if clear_workspace:
        root = pathlib.Path(config['root'])
        signac_rc = root / 'signac.rc'
        workspace = root / 'workspace'
        if not signac_rc.exists():
            raise FileNotFoundError(f"Cannot find signac.rc in {root}")
        if not workspace.exists():
            raise FileNotFoundError(f"Cannot find workspace in {root}")

        answer = None
        while answer is None or answer.lower() not in ['yes', '']:
            answer = input(
                'Are you sure you want to clear the workspace? (yes/N) ')
        if answer == 'yes':
            print(f'Clearing workspace {workspace}')
            shutil.rmtree(str(workspace))
            print(f'Clearing other files in {root}')
            for file in root.iterdir():
                os.remove(file)
        sys.exit()

    # initialize project
    project: signac.Project = signac.init_project("phase-diagrams",
                                                  root=config["root"])

    statepoint_grid_ka_lj = {
        "it": range(3),
        "A_frac": [80, 70, 60, 50],
        "delta": [None, 0.1, 0.2, 0.3, 0.4, 0.5]
    }

    if "cur_seed" not in project.doc:
        project.doc["cur_seed"] = 0

    for sp in grid(statepoint_grid_ka_lj):
        universal = dict(N=num, p=pressure)
        sp.update(universal)
        job = project.open_job(statepoint=sp)
        job.init()
        if "init" not in job.doc:
            print(
                f"Initializing job {job.id} with seed {project.doc['cur_seed']}"
            )
            job.document["seed"] = project.doc["cur_seed"]
            job.document['init'] = True
            project.doc["cur_seed"] += 1

    # symlink the project directory to the current folder
    try:
        os.symlink(config["root"], "root")
    except FileExistsError:
        if os.path.islink(config["root"]) and os.path.realpath(
                config["root"]) != os.path.realpath("root"):
            os.remove("root")
            os.symlink(config["root"], "root")


if __name__ == "__main__":
    main()
