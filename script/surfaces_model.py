import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import re
import numpy as np
from scipy.interpolate import griddata
import math, sys 

N = 1000
C = 1000

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
    for h in range(0.1, 1, 0.05):
        for w in range(0.1, 1, 0.05):
            if name == "ESS":
                expr = C / (h * w)
            elif name == "LRW":
                expr = 1/h + 1/w + (2*h + w) / C
            elif name == "TSS":
                pass
            elif name == "EUC":
                pass
            elif name == "MOON":
                pass
            elif name == "TLI":
                pass
            elif name == "WMC":
                pass
            elif name == "MHCF":
                pass
            r.append((h, w, expr))
    return r

def execute_models(combinations, labels):
    
    data = models(C / h * w)

    # Separate the dimensions and costs
    dimensions = np.array([point[:2] for point in data])
    costs = np.array([point[2] for point in data])

    find_best_global_solution(dimensions, costs, labels[i])
    
    output_filename = f"surface_plot_combination_{i+1}.png"
    create_surface_plot(dimensions, costs, combination, output_filename, labels[i])

if __name__ == "__main__":

    # Create surface plots for each combination of dimensions
    combinations = [(0, 1)]
    labels = [("h", "w")]

    execute_models(combinations, labels)