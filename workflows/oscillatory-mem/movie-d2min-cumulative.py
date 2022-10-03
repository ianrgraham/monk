from schmeud import qlm
import gsd.hoomd
import freud

import matplotlib.pyplot as plt

import numpy as np
from numpy.linalg import inv
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
norm = colors.Normalize(vmin=1e-2, vmax=2.0)

file = "/media/ian/Data2/monk/oscillatory-mem/view/dumps/40/max_strain/0.08/it/609/job/traj.gsd"

traj = gsd.hoomd.open(file)

snap: gsd.hoomd.Snapshot = traj[0]

def wrap(x, box):
    return 

def d2min_py(b0, b):
    """Calculates D2min for a set of bonds
    Args
        b0: initial bond lengths
        b: final bond lengths
    """
    V = b0.transpose().dot(b0)
    W = b0.transpose().dot(b)
    J = inv(V).dot(W)
    non_affine = b0.dot(J) - b
    d2min = np.sum(np.square(non_affine))
    return d2min

def d2min_slow(snap, snap_later, nlist, box, box_later):
    d2min = []
    for i, (head, nn) in enumerate(zip(nlist.segments, nlist.neighbor_counts)):
        indices = nlist.point_indices[head:head+nn]
        b0 = box.wrap(snap.particles.position[indices]
                    - snap.particles.position[i])[:, :2]
        b = box_later.wrap(snap_later.particles.position[indices]
                    - snap_later.particles.position[i])[:, :2]
        d2min.append(d2min_py(b0, b))
    return np.array(d2min)

pos = []
rad = snap.particles.diameter[:] * 12.0
color = []


for i in range(0, 1):
    snap = traj[i*40]
    box = snap.configuration.box
    fbox = freud.box.Box.from_box(box)
    for j in range(40):
        snap_later = traj[i*40 + j]
        box_later = snap_later.configuration.box
        fbox_later = freud.box.Box.from_box(box_later)

        pos2d = np.ascontiguousarray(snap_later.particles.position[:, :2])
        pos2d[:, 0] /= box[0]*0.6
        pos2d[:, 1] /= box[1]*0.6
        pos.append(pos2d)

        nlist_query = freud.locality.LinkCell.from_system(snap)
        nlist = nlist_query.query(snap.particles.position, {'num_neighbors': 20}).toNeighborList()

        d2min = dynamics.d2min_frame(snap.particles.position[:, :2], snap_later.particles.position[:, :2], nlist.query_point_indices, nlist.point_indices, (box, box_later))

        # d2min = d2min_slow(snap, snap_later, nlist, fbox, fbox_later)

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

from PIL import Image

def render_snap(self, uri, format=None, writer_kwargs={}):
    im = self.render()
    im = Image.fromarray(im)
    im.save(uri, format=format, **writer_kwargs)


render_snap(sys, "test.png")


# sys.run_app()
# sys.render_imageio("d2min-cumulative.mov", len(pos))