import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import re
import numpy as np
from scipy.interpolate import griddata
import math, sys 

n = 1000 # size of 2D array 
C = 1024 # Cache's size
l = 1024 # Cache Line size 

def create_surface_plot(dimensions, costs, combination, output_filename, label):
    x_idx, y_idx = (0, 1)

    # Create a grid for the selected dimensions
    x_range = np.linspace(dimensions[:, x_idx].min(), dimensions[:, x_idx].max(), num=100)
    y_range = np.linspace(dimensions[:, y_idx].min(), dimensions[:, y_idx].max(), num=100)
    x_grid, y_grid = np.meshgrid(x_range, y_range)

    # Interpolate the Z values (cost) for the grid
    z_grid = griddata(dimensions[:, (x_idx, y_idx)], costs, (x_grid, y_grid), method="cubic")

    # Set up the 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Plot the surface
    surface = ax.plot_surface(x_grid, y_grid, z_grid, cmap="viridis", linewidth=0, antialiased=False)

    # Add color bar for the costs
    cbar = plt.colorbar(surface)
    cbar.set_label("Cost")

    # Set the axis labels
    ax.set_xlabel(label[0])
    ax.set_ylabel(label[1])
    ax.set_zlabel("Cost")

    # Save the plot as a PNG image
    plt.savefig(output_filename)
    plt.show()
    plt.close(fig)

def models(name):

    r = []
    for h in np.arange(0.1, 1, 0.05):
        for w in np.arange(0.1, 1, 0.05):
            if name == "ESS":
                expr = C / (h * w)
            elif name == "LRW":
                expr = 1/h + 1/w + (2*h + w) / C
            elif name == "TSS":
                expr = (2*h + w)/h * w
            elif name == "EUC":
                expr = 1/h + 1/w
            elif name == "MOON":
                expr = 1/h + 1/w + (h+w)/C
            elif name == "TLI":
                expr = 1/h + 1/w + (h+w)/C + h*w/(C**2)
            elif name == "WMC":
                expr = C/h * w
            elif name == "MHCF":
                expr = (1/h+1/w) * (1/n + 1/l) + 2/(h*w)
            else:
                print("Model not exist!")
                exit(0)
            r.append((h, w, expr))
    return r

def execute_models(combination, label, name):
    
    data = models(name)

    # Separate the dimensions and costs
    dimensions = np.array([point[:2] for point in data])
    costs = np.array([point[2] for point in data])
    
    output_filename = f"surface_plot_{name}.png"
    create_surface_plot(dimensions, costs, combination, output_filename, label)

if __name__ == "__main__":

    name_model = ["ESS", "LRW", "TSS", "EUC", "MOON", "TLI", "WMC", "MHCF"]

    combinations = [(0, 1)]
    labels = [("h", "w")]

    for name in name_model:
        execute_models(combinations[0], labels[0], name)