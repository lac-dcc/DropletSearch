from mpl_toolkits import mplot3d
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
import sys, math
import sys, math
import numpy as np

def read_file(csv_file):

    f = open(csv_file, "r")
    
    size = 0
    lista = []
    c = 0
    for l in f.readlines():
        l = l.strip().replace(" ", "").split(",")
        lista.append(l)
        size += 1
    
    dim = int(math.sqrt(size))
    m = [[0 for col in range(dim)] for row in range(dim)]

    for l in lista:
        row = int(l[0])//4
        col = int(l[1])//4

        value = l[2]
        
        m[row][col] = value

    for i in range(dim):
        for j in range(dim):
            print(m[i][j], ",", end="")
        print()
    
    x = [[row*4 for col in range(dim)] for row in range(dim)]

    return np.array(x), np.array(x).T, np.array(m)

if __name__ == "__main__":

    csv_file = sys.argv[1]
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