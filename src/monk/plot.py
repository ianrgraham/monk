import matplotlib.pyplot as plt

def scalar_quantity(traj, quantity_name):

    fig, ax = plt.subplots()

    timestep = []
    walltime = []
    quantity = []

    for frame in traj:
        timestep.append(frame.configuration.step)
        quantity.append(
            frame.log[quantity_name][0])

    ax.plot(timestep, quantity)

    return (fig, ax), (timestep, quantity)