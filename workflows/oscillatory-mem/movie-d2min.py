from schmeud import qlm
import gsd.hoomd
import freud

import matplotlib.pyplot as plt

import numpy as np
import itertools

import matplotlib as mpl
from matplotlib import cm, colors

from monk import workflow
import signac
from schmeud._schmeud import dynamics
from noovpy import system
import vispy

vispy.use("Glfw")

cmap = cm.jet
norm = colors.Normalize(vmin=1e-2, vmax=1.0)

file = "/media/ian/Data2/monk/oscillatory-mem/view/dumps/40/max_strain/0.08/it/609/job/traj.gsd"

traj = gsd.hoomd.open(file)

snap: gsd.hoomd.Snapshot = traj[0]

pos = []
rad = snap.particles.diameter[:] * 12.0
color = []

end = 40*20

for i in range(0, end - 1):
    snap = traj[i]
    snap_later = traj[i + 1]
    box = snap.configuration.box
    box_later = snap_later.configuration.box

    pos2d = np.ascontiguousarray(snap.particles.position[:, :2])
    pos2d[:, 0] /= box[0]*0.6
    pos2d[:, 1] /= box[1]*0.6
    pos.append(pos2d)

    nlist_query = freud.locality.LinkCell.from_system(snap)
    nlist = nlist_query.query(snap.particles.position, {'num_neighbors': 20}).toNeighborList()

    d2min = dynamics.d2min_frame(snap.particles.position[:, :2], snap_later.particles.position[:, :2], nlist.query_point_indices, nlist.point_indices, (box, box_later))

    # bins = np.geomspace(np.min(d2min), np.max(d2min), 20)
    # plt.hist(d2min, bins=bins)
    # plt.xscale('log')
    # plt.show()
    # raise ValueError

    color.append(np.ascontiguousarray(cmap(norm(d2min)).astype(np.float32)[:,:3]))



sys = system.System(
    0.05,
    {
        "size": (1000, 1000),
        "resizable": False,
        "autoswap": True,
        "px_scale": 1
    }
)

sys.add(
    system.Particles(
        pos,
        rad,
        color
    )
)

# sys.run_app()
sys.render_imageio("d2min.mov", len(pos))