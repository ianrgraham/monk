# %%
import numpy as np

from hoomd.neb_plugin import neb
import hoomd
import freud
import gsd.hoomd
import monk
from monk import prep, pair, render

import multiprocessing as mp
import multiprocessing.pool as mpp
import multiprocessing.shared_memory as sm
import zmq

# %%
def make_snap(box, pos):
    snap = gsd.hoomd.Snapshot()
    snap.configuration.box = list(box.L) + [0, 0, 0]
    snap.configuration.dimensions = 2
    snap.particles.N = len(pos)
    snap.particles.position = pos
    snap.particles.types = ["A"]

    return snap

# %%
# prepare lattice with a defect

Nx = 5

snap_box, snap_pos = freud.data.UnitCell.hex().generate_system((Nx, Nx, 1), scale=0.8)

future_pos = np.copy(snap_pos)
mid = future_pos[Nx*Nx,:].copy()
future_pos[Nx*Nx,:] = future_pos[0,:]
future_pos[Nx*(Nx+1),:] = mid

snap_pos = np.delete(snap_pos, 0, axis=0)
future_pos = np.delete(future_pos, 0, axis=0)

snap = make_snap(snap_box, snap_pos)
future_snap = make_snap(snap_box, future_pos)

sim = hoomd.Simulation(device=hoomd.device.CPU())
sim.create_state_from_snapshot(snap)

# %%
future_hoomd_snap = hoomd.Snapshot.from_gsd_snapshot(future_snap, sim.device.communicator)

# %%
render.render_disk_frame(sim.state.get_snapshot(), Nx*2)

# %%
render.render_disk_frame(future_hoomd_snap, Nx*2)

# %%
def setup_node(sim, k):
    neb_integrator = neb.NEB(0.01, 1e-3, 1e-3, 1e-3, k=k)
    nlist = hoomd.md.nlist.Cell(0.3)
    hertzian = pair.bi_hertz(nlist)
    neb_integrator.forces = [hertzian]
    nve = hoomd.md.methods.DisplacementCapped(hoomd.filter.All(), 0.01)
    neb_integrator.methods = [nve]

    sim.operations.integrator = neb_integrator

    sim.run(0)

def make_node(box, pos, device=hoomd.device.CPU(), k=1.0):
    sim = hoomd.Simulation(device=device)
    snap = make_snap(box, pos)
    sim.create_state_from_snapshot(snap)
    setup_node(sim, k)
    return sim

def couple_neb_minimizers(sims):

    for i in range(len(sims)):
        minimizer = sims[i].operations.integrator
        assert isinstance(minimizer, neb.NEB)

    for i in range(len(sims)-1):
        left_minimizer = sims[i].operations.integrator
        assert isinstance(left_minimizer, neb.NEB)
        right_minimizer = sims[i+1].operations.integrator
        assert isinstance(right_minimizer, neb.NEB)
        left_minimizer.couple_right(right_minimizer)
        right_minimizer.couple_left(left_minimizer)


# %%
neb_sims = []

images = 20

neb_sims.append(make_node(snap_box, snap_pos))
for i in range(images):
    f = float(i)/float(images+1)
    neb_sims.append(make_node(snap_box, snap_pos*(1-f) + future_pos*f))
neb_sims.append(make_node(snap_box, future_pos))

couple_neb_minimizers(neb_sims)

# %%
render.render_disk_frame(neb_sims[10].state.get_snapshot(), Nx*2)

# %%
def run(sim: hoomd.Simulation):
    sim.run(1000, release_gil=True)

with mpp.ThreadPool(images+2) as pool:
    print(pool.map(run, neb_sims))

# %%
render.render_disk_frame(neb_sims[11].state.get_snapshot(), Nx*2)

# %%



