from gsd.hoomd import HOOMDTrajectory, _HOOMDTrajectoryView, _HOOMDTrajectoryIterable
import freud

from typing import Union

import numpy as np


Trajectory = Union[HOOMDTrajectory, _HOOMDTrajectoryView, _HOOMDTrajectoryIterable]


def unwrap_traj_disp_const(traj: Trajectory) -> np.ndarray:
    """Unwrap trajectory dispalcements assuming a constant box."""
    
    snap0 = traj[0]
    box = freud.box.Box.from_box(snap0.configuration.box)
    pos0 = snap0.particles.position
    out = np.zeros((len(traj), *pos0.shape), dtype=np.float32)
    image = np.zeros_like(pos0, dtype=np.int32)
    prev = pos0
    for idx, snap in enumerate(traj):
        pos = snap.particles.position
        image -= box.get_images(pos - prev)
        out[idx] = box.unwrap(pos, image) - pos0
        prev = pos
    return out


def unwrap_traj_disp_affine(traj: Trajectory) -> np.ndarray:
    """Unwrap trajectory dispalcements by applying affine transforms of variable boxes."""
    snap0 = traj[0]
    box0 = freud.box.Box.from_box(snap0.configuration.box)
    pos0 = box0.make_fractional(snap0.particles.position)
    unit_box = freud.box.Box.cube(1)
    prev = pos0
    image = np.zeros((len(pos0), 3), dtype=np.int32)
    out = np.zeros((len(traj), *pos0.shape), dtype=np.float32)
    for frame in traj:
        box = freud.box.Box.from_box(frame.configuration.box)
        pos = box.make_fractional(frame.particles.position)
        image -= unit_box.get_images(pos - prev)
        x = box.make_absolute(pos)
        x = box.unwrap(x, image)
        x -= box.make_absolute(pos0)
        out.append(x)
        prev = pos
    return out
