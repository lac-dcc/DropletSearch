import sys, math
import numpy as np

if __name__ == "__main__":

    csv_file = sys.argv[1]

    f = open(csv_file, "r")
    
    size = 0
    lista = []
    for l in f.readlines():
        l = l.strip().replace(" ", "").split(",")
        lista.append(l)
        size += 1
    
    dim = int(math.sqrt(size))
    m = [[0 for col in range(dim)] for row in range(dim)]

    for l in lista:
        row = int(l[1])//8
        col = int(l[2])//8
        
        value = []
        for i in range(3,len(l)):
        	value.append(float(l[i]))
        
        m[row][col] = np.array(value).mean()

    for i in range(dim):
        for j in range(dim):
            print(m[i][j], ",", end="")
        print()
