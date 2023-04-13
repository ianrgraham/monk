import numpy as np
import fresnel
import pathlib
import os

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from matplotlib import colors, cm

import pandas as pd
import gsd.hoomd
import signac
import PIL.Image

from monk import nb, prep, pair, render, utils, workflow, grid

parent = pathlib.Path(os.getcwd()).parent / "config.yaml"
config = workflow.get_config(parent.as_posix())
project: signac.Project = signac.get_project(config['root'])

example_job = list(project.find_jobs({"delta": 0.0}))[0]
filename = example_job.fn("short_runs/temp-0.45/fire_traj.gsd")
soft_file = example_job.fn("short_runs/temp-0.45/struct-descr.parquet")
df = pd.read_parquet(soft_file)
gsd_file = gsd.hoomd.open(name=filename, mode="rb")

cmap = cm.jet
norm = mpl.colors.Normalize(vmin=df["softness"].min(), vmax=df["softness"].max())

for idx in sorted(df["frame"].unique())[:10]:
    idx = int(idx)
    snap = gsd_file[idx]

    box = snap.configuration.box
    pos = snap.particles.position
    cond = (pos[:, 0] < 0.0) | (pos[:, 1] < 0.0) | (pos[:, 2] < 0)
    N = len(pos)
    # N = len(pos[cond])
    particle_types = snap.particles.typeid
    colors = np.empty((N, 3))
    diams = snap.particles.diameter

    data = df[df.frame == idx]
    idxs = data.tag.to_numpy()

    

    # # Color by typeid
    colors[data.tag.to_numpy()] = cmap(norm(data["softness"].to_numpy()))[:,:3] # A type
    colors[particle_types == 1] = fresnel.color.linear([0, 0, 0]) # B type

    # pos = pos[cond]
    colors = colors[cond]
    diams = diams[cond]
    N = len(pos[cond])

    light = fresnel.light.Light((1, 1, 1))
    fill_light = fresnel.light.Light((0, 0, 1), color=(0.2, 0.2, 0.2), theta=3.141592)
    scene = fresnel.Scene(lights=[light, fill_light])

    # Spheres for every particle in the system
    geometry = fresnel.geometry.Sphere(scene, N=N, radius=diams/2)
    geometry.position[:] = pos[cond]
    geometry.material = fresnel.material.Material(roughness=1.0)
    geometry.outline_width = 0.05

    # use color instead of material.color
    geometry.material.primitive_color_mix = 1.0
    geometry.color[:] = fresnel.color.linear(colors)

    scene.camera = fresnel.camera.Orthographic.fit(scene)
    image = fresnel.pathtrace(scene, w=1000, h=1000, light_samples=10)
    image = image[:]
    # print(image.shape)
    shape = image.shape
    for i in range(shape[0]):
        for j in range(shape[1]):
            # print(image[i, j, 3])
            if image[i, j, 3] < 255:
                image[i, j, :3] = image[i, j, 3]*image[i, j, :3]/255 + (255-image[i, j, 3])*np.array([255, 255, 255])/255
                image[i, j, 3] = 255
    # image = fresnel.preview(scene)

    pil_image = PIL.Image.fromarray(image, mode='RGBA')
    pil_image.save(f'2test-{idx:03d}.png')