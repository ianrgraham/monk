from typing import Optional
import matplotlib.pyplot as plt

from gsd.hoomd import Snapshot
from matplotlib import axes

def scalar_quantity(
    traj: Snapshot,
    quantity_name: str,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    ax: Optional[axes.Axes] = None
):

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    timestep = []
    quantity = []

    for frame in traj:
        timestep.append(frame.configuration.step)
        quantity.append(
            frame.log[quantity_name][0])

    ax.plot(timestep, quantity)

    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)

    return (fig, ax), (timestep, quantity)