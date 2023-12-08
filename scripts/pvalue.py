# Import libraries
import numpy as np
import sys, math
from scipy import stats
 
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
    m = [[0 for col in range(dim)] for row in range(dim)]

    for l in lista:
        row = int(l[1])//8
        col = int(l[2])//8
        values = []
        for i in range(3,len(l)):
            values.append(float(l[i]))
        m[row][col] = values

    return m

def pvalue(m, row_1, col_1, row_2, col_2):

    data_1 = np.array(m[row_1][col_1])
    data_2 = np.array(m[row_2][col_2])

    p_value = stats.ttest_ind(data_1, data_2)

    print(p_value)

if __name__ == "__main__":

    if len(sys.argv) > 5:
        csv_file = sys.argv[1]
        row_1 = int(sys.argv[2]) // 8
        col_1 = int(sys.argv[3]) // 8
        row_2 = int(sys.argv[4]) // 8
        col_2 = int(sys.argv[5]) // 8
    else:
        print("Example: python3 pvalue.py <FILE.csv> <ROW_1> <COL_1> <ROW_2> <COL_2>")
        exit(0)
    
    m = read_file(csv_file)

    pvalue(m, row_1, col_1, row_2, col_2)
    


    
