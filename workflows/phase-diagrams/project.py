"""Workflow to automatically build the glassy phase diagram for """

import flow
import signac
import hoomd
import numpy as np
import yaml
import sys

from monk import project_path, project_view, safe_clean_signac_project, grid
from monk import pair, prep, methods

with open("config.yaml") as file:
    CONFIG = yaml.load()
    WORKSPACE = CONFIG["workspace"]

print(WORKSPACE)

sys.exit()

class Project(flow.FlowProject):
    pass

@Project.operation
@Project.pre.true('init')
@Project.post.true('init_state')
def init_state(job: signac.Project.Job):
    N = job.sp["N"]
    phi = job.sp["phi"]
    seed = job.doc["seed"]

    device = hoomd.device.auto_select()
    sim = hoomd.Simulation(device, seed=seed)

    rng = prep.init_rng(seed)
    L = prep.len_from_phi(N, phi, dim=2)
    snap = prep.uniform_random_snapshot(
        N, L, rng, dim=2, diams=[7/6, 5/6]
    )
    sim.create_state_from_snapshot(snap)

    hoomd.write.GSD.write(sim.state, job.fn("init.gsd"))

    job.doc["init_state"] = True