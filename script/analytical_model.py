Ni = 10000
Nj = 10000
Nk = 10000
Ti_0=1   
Tj_0=255   
Tk_0=255   
Ti_1=386   
Tj_1=255   
Tk_1=255

'''
dims: [i, j, k]

arrays:
   A: [[i, k], 1]
   B: [[k, j], 1]
   C: [[i, j], 2]

reuse:
   A: [j]
   B: [i]
   C: [k]

values:
  n_levels: 2
  cache: [65536, 262144]
  dims: [10000, 10000, 10000]
'''

def find_best_global_solution(dimensions, costs, label):
    best_cost = 100000000000000
    best_dimensions = (0, 0)
    for i in range(len(costs)):
        if best_cost > costs[i]:
            print(best_cost, dimensions[i])
            best_cost = costs[i]
            best_dimensions = dimensions[i]
    print("%s, %d, %s, %d, cost = %d" %(label[0], best_dimensions[0], label[1], best_dimensions[1], best_cost))


def bench_cacheL1(combinations):
    Ni, Nj, Nk = 10000, 10000, 10000
    Tj_0, Ti_0 = 255, 255
    print(combinations)
    r = []
    for i in range(1,256,5):
        for j in range(1,256,5):
            Tj_0, Ti_0 = i, j
            expr = Ni*Nj*Nk*(1/Tj_0 + 1/Ti_0 + 2/Nk)
            r.append((i, j, expr))
    return r

def bench_cacheL2(combinations):
    Ni, Nj, Nk = 10000, 10000, 10000
    Tj_0, Tk_0, Ti_1, Tj_1 = 255, 255, 386, 255  
    print(combinations)
    r = []
    for i in range(1,8192,1024):
        for j in range(1,8192,1024):
            if combinations[0] == 0 and combinations[1] == 1:
                Tj_0, Tk_0 = i, j
            elif combinations[0] == 0 and combinations[1] == 2:
                Tj_0, Ti_1 = i, j
            elif combinations[0] == 0 and combinations[1] == 3:
                Tj_0, Tj_1 = i, j
            elif combinations[0] == 1 and combinations[1] == 2:
                Tk_0, Ti_1 = i, j
            elif combinations[0] == 1 and combinations[1] == 3:
                Tk_0, Tj_1 = i, j
            elif combinations[0] == 2 and combinations[1] == 3:
                Ti_1, Tj_1 = i, j 
            expr = Ni*Nj*Nk*(1/Tj_1 + 1/Ti_1 + 2/Nk) + Ni*Nj*Nk*(2/Tk_0 + 1/Tj_0 + 1/Ti_1)
            r.append((i, j, expr))
    return r

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import re
import numpy as np
from scipy.interpolate import griddata
import math, sys 

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


def execute_benchmark(cache, combinations, labels):
    for i, combination in enumerate(combinations):
        if cache == "L1":
            data = bench_cacheL1(combination)
        else:
            data = bench_cacheL2(combination)
        
        # Separate the dimensions and costs
        dimensions = np.array([point[:2] for point in data])
        costs = np.array([point[2] for point in data])

        find_best_global_solution(dimensions, costs, labels[i])
        
        output_filename = f"surface_plot_combination_{i+1}.png"
        create_surface_plot(dimensions, costs, combination, output_filename, labels[i])

if __name__ == "__main__":

    # Create surface plots for each combination of dimensions
    combinations = [(0, 1),(0, 2),(0, 3),(1, 2),(1, 3),(2, 3)]
    labels = [("Tj_0", "Tk_0"), ("Tj_0", "Ti_1"), ("Tj_0", "Tj_1"), ("Tk_0", "Ti_1"), ("Tk_0", "Tj_1"), ("Ti_1", "Tj_1")]
    #execute_benchmark("L2", combinations, labels)
    
    combinations = [(0, 1)]
    labels = [("Tj_0", "Ti_0")]
    execute_benchmark("L1", combinations, labels)
