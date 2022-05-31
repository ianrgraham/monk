"""Workflow to test the dynamical and structural properties of various systems """

import flow
import hoomd
import numpy as np
import signac
import os.path

import argparse

from monk import project_path, project_view, safe_clean_signac_project, grid
from monk import pair, prep, methods

class Project(flow.FlowProject):

    def _main_init(self):
        
        project = self
        
        if "initialized" not in project.doc:
            print("Initializing project")
            project.doc["initialized"] = True
        else:
            raise RuntimeError("Project has already been initialized")

        if "cur_seed" not in project.doc:
            project.doc["cur_seed"] = 0

        # Initialize the data space

        statepoint_grid_ka_lj = {
            "it": range(5), 
            "phi": [1.0, 1.1, 1.15, 1.2, 1.3],
            "A_frac": [80, 75, 70, 65, 60]
        }

        for sp in grid(statepoint_grid_ka_lj):
            universal = dict(N=512, init_kT=1.4, final_kT=0.4, dt=2.5e-3, steps=1_000_000, equil_steps=100_000, dumps=10)
            sp = sp.update(universal)
            job = project.open_job(sp).init()
            if "init" not in job.doc:
                job.document["seed"] = project.doc["cur_seed"]
                job.document['init'] = True
                project.doc["cur_seed"] += 1

    def _main_clear(self):
        safe_clean_signac_project(self.root_directory(), prepend_monk=False)

    def main(self):

        parser = argparse.ArgumentParser()

        subparsers = parser.add_subparsers()

        parser_init = subparsers.add_parser(
            "init",
            description="Initialize statepoints for processing.",
        )
        parser_init.set_defaults(func=self._main_init)

        parser_clear = subparsers.add_parser(
            "clear",
            description="Clear current state points",
        )
        parser_clear.set_defaults(func=self._main_clear)

        super().main(parser=parser)


@Project.operation
@Project.pre.true('init')
@Project.post.true('init_state')
def init_state(job: signac.Project.Job):
    N = job.sp["N"]
    phi = job.sp["phi"]
    seed = job.doc["seed"]
    A_frac = job.sp["A_frac"]

    device = hoomd.device.auto_select()
    sim = hoomd.Simulation(device, seed=seed)

    

    rng = prep.init_rng(seed)
    L = prep.len_from_phi(N, phi)
    snap = prep.approx_euclidean_snapshot(
        N, L, rng, dim=3, particle_types=["A", "B"], ratios=[A_frac, 100-A_frac], diams=[1.0, 0.88]
    )
    sim.create_state_from_snapshot(snap)

    hoomd.write.GSD.write(sim.state, job.fn("init.gsd"))

    job.doc["init_state"] = True


@Project.operation
@Project.pre.after(init_state)
@Project.post.true('simulated')
def run_nvt_sim(job: signac.Project.Job):
    seed = job.doc["seed"]
    init_kT = job.sp["init_kT"]
    final_kT = job.sp["final_kT"]
    dt = job.sp["dt"]
    equil_steps = job.sp["equil_steps"]
    steps = job.sp["steps"]
    dumps = job.sp["dumps"]

    device = hoomd.device.auto_select()
    sim = hoomd.Simulation(device, seed=seed)

    sim.create_state_from_gsd(job.fn("init.gsd"))

    integrator = hoomd.md.Integrator(dt=dt)
    nlist = hoomd.md.nlist.Tree(0.2)
    pot_pair = pair.KA_LJ(nlist)
    integrator.forces.append(pot_pair)

    tstart = sim.timestep + equil_steps
    tramp = tstart + steps

    kT_variant = hoomd.variant.Ramp(init_kT, final_kT, tstart, tramp)

    nvt = hoomd.md.methods.NVT(
        kT=kT_variant,
        filter=hoomd.filter.All(),
        tau=0.5
    )

    integrator.methods.append(nvt)
    sim.operations.integrator = integrator

    sim.run(0)
    nvt.thermalize_thermostat_dof()

    sim.run(equil_steps)

    gsd_writer = hoomd.write.GSD(
        filename=job.fn("traj.gsd"),
        trigger=hoomd.trigger.Periodic(steps/dumps, phase=sim.timestep),
        mode='wb',
        filter=hoomd.filter.All(),
    )

    sim.operations.writers.append(gsd_writer)

    sim.run(steps+1)

    job.doc["simulated"] = True

@Project.operation
@Project.pre.after(run_nvt_sim)
@Project.post.true('validated')
def validate(job: signac.Project.Job):
    pass


project: Project = Project.init_project(name="GenGlassStates3D", root=project_path("initial-configs/3d-glass/param_explor"))

if __name__ == '__main__':
    project.main()