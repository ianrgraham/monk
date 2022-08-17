import flow
import hoomd
import numpy as np
import signac
import os.path

from monk import project_path, project_view, safe_clean_signac_project, grid
from monk import pair, prep


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
    L = prep.len_from_phi(N, phi)
    snap = prep.approx_euclidean_snapshot(N,
                                          L,
                                          rng,
                                          dim=3,
                                          particle_types=["A", "B"],
                                          ratios=[80, 20],
                                          diams=[1.0, 0.88])
    sim.create_state_from_snapshot(snap)

    hoomd.write.GSD.write(sim.state, job.fn("init.gsd"))
    job.doc["init_state"] = True


class LogTrigger(hoomd.trigger.Trigger):

    def __init__(self, base, start, step, ref):
        self.ref = ref
        self.base = base
        self.step = step
        self.cidx = start
        self.cstep = self.base**self.cidx
        super().__init__()

    def compute(self, timestep):
        result = False
        while timestep - self.ref > self.cstep:
            result = True
            self.cidx += self.step
            self.cstep = self.base**self.cidx
        return result


@Project.operation
@Project.pre.after(init_state)
@Project.post.true('done')
def run_nvt_sim(job: signac.Project.Job):
    seed = job.doc["seed"]
    kT = job.sp["kT"]
    dt = job.sp["dt"]
    equil_steps = job.sp["equil_steps"]
    steps = job.sp["steps"]

    device = hoomd.device.auto_select()
    sim = hoomd.Simulation(device, seed=seed)

    sim.create_state_from_gsd(job.fn("init.gsd"))

    integrator = hoomd.md.Integrator(dt=dt)
    nlist = hoomd.md.nlist.Tree(0.2)
    pot_pair = pair.KA_LJ(nlist)
    integrator.forces.append(pot_pair)

    nvt = hoomd.md.methods.NVT(kT=kT, filter=hoomd.filter.All(), tau=0.5)
    integrator.methods.append(nvt)
    sim.operations.integrator = integrator

    sim.run(0)

    nvt.thermalize_thermostat_dof()

    print("Everything is set up")

    sim.run(equil_steps)

    print("Writing to disk")

    gsd_writer = hoomd.write.GSD(
        filename=job.fn("traj.gsd"),
        trigger=LogTrigger(10, 1, 0.1, sim.timestep),
        mode='wb',
        filter=hoomd.filter.All(),
    )

    sim.operations.writers.append(gsd_writer)

    sim.run(steps)

    job.doc["done"] = True


project: Project = Project.init_project(name="GenerateEquilibratedStates",
                                        root=project_path("test-signac-gen"))

# safe_clean_signac_project("test-signac-gen")
# exit()

if "seed" not in project.doc:
    project.doc["seed"] = 0

# Initialize the data space
seed = sum([int("init" in job.document) for job in project])
iterations = 5
for kT in np.linspace(1.0, 2.0, 5):
    for it in range(iterations):
        sp = dict(N=512,
                  phi=1.2,
                  kT=float(kT),
                  dt=1e-3,
                  steps=100_000,
                  equil_steps=100_000,
                  iter=it)
        job = project.open_job(sp).init()
        if "init" not in job.doc:
            print("Hey!")
            job.document["seed"] = project.doc["seed"]
            job.document['init'] = True
            project.doc["seed"] += 1

if __name__ == '__main__':
    project.main()
