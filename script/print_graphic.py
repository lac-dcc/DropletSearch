# Import libraries
from mpl_toolkits import mplot3d
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
import sys, math
 
def read_file(csv_file):
    f = open(csv_file, "r")

    row, col = [], []

    size = 0
    lista = []
    for l in f.readlines():
        l = l.strip().replace(" ", "").split(",")
        lista.append(l)
        size += 1
    
    dim = int(math.sqrt(size))
    value = [[0 for col in range(dim)] for row in range(dim)]

    for l in lista:
        row = int(l[1])//8
        col = int(l[2])//8
        
        value[row][col] = float(l[3])
    
    x = [[row*8 for col in range(dim)] for row in range(dim)]

    return np.array(x), np.array(x).T, np.array(value)

if __name__ == "__main__":

    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
    else:
        print("Example: python3 print_graphic.py")
        exit(0)

    x, y, z = read_file(csv_file)

    # Creating figure
    fig = plt.figure()
    ax = plt.axes(projection ='3d')

    # Creating color map
    my_cmap = "RdYlGn"
    
    # Creating plot
    surf = ax.plot_surface(x, y, z, cmap=my_cmap, edgecolor ='none', antialiased=False)
    
    fig.colorbar(surf, ax = ax, shrink = 0.5, aspect = 5)
    
    # show plot
    plt.show()
