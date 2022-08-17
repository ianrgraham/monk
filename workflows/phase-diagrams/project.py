"""Workflow to automatically build glassy phase diagrams.

Statepoints here define isobaric quenches that should begin in the liqiud state and terminate in the glass.
Configurations are thermalized until a target diffusivity is reached. From there, the system temperature is
cooled with a dynamic update scheme until the t


"""

import flow
import signac
import hoomd
import numpy as np
import yaml
import sys

from monk import project_path, project_view, safe_clean_signac_project, grid
from monk import pair, prep, methods, workflow

config = workflow.get_config()


class Project(flow.FlowProject):
    pass


@Project.operation
@Project.pre.true('init')
@Project.post.true('init_state')
def init_state(job: signac.Project.Job):
    N = job.sp["N"]
    init_phi = job.sp["init_phi"]
    target_D = job.sp["target_D"]
    seed = job.doc["seed"]
    A_frac = job.sp["A_frac"]

    device = hoomd.device.auto_select()
    sim = hoomd.Simulation(device, seed=seed)

    rng = prep.init_rng(seed)
    L = prep.len_from_phi(N, init_phi)
    snap = prep.approx_euclidean_snapshot(N,
                                          L,
                                          rng,
                                          dim=3,
                                          particle_types=["A", "B"],
                                          ratios=[A_frac, 100 - A_frac],
                                          diams=[1.0, 0.88])
    sim.create_state_from_snapshot(snap)

    hoomd.write.GSD.write(sim.state, job.fn("init.gsd"))

    job.doc["init_state"] = True


# TODO add operation to map out the dynamics

# TODO once we understand the rough dynamics of each system, we can
