"""Script to generate statepoints of the workspace"""

import os

import signac
from monk import workflow, grid

# grabs the project configuration
config = workflow.get_config()

# initialize project
project = signac.get_project(root=config["root"])

statepoint_grid_ka_lj = {
    "it": range(3),
    "A_frac": [80, 70, 60, 50],
    "delta": [None, 0.1, 0.2, 0.3, 0.4, 0.5]
}

for sp in grid(statepoint_grid_ka_lj):
    universal = dict(N=16384, p=1.0)
    sp = sp.update(universal)
    job = project.open_job(sp).init()
    if "init" not in job.doc:
        job.document["seed"] = project.doc["cur_seed"]
        job.document['init'] = True
        project.doc["cur_seed"] += 1

# symlink the project directory to the current folder
os.symlink(config["root"], "root")