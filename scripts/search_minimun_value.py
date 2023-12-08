# Import libraries
import numpy as np
import sys, math
import pandas as pd
from scipy import stats

'''
'''
def processing_input(csv_file : str):
    df = pd.read_csv(csv_file)
    df.pop("sum")

    m = {}
    for l in df.values:
        a = int(l[0])
        b = int(l[1])
        if a in m:
            m[a][b] = np.array(l[2:])
        else:
            m[a] = {}
            m[a][b] = np.array(l[2:])
    return m
    

def pvalue(data:dict, pos_i:int, pos_j:int, curr_i:int, curr_j:int):
    data_1 = np.array(data[pos_i][pos_j])
    data_2 = np.array(data[curr_i][curr_j])
    if len(data_1) <= 1:
    	return 0
    return stats.ttest_ind(data_1, data_2).pvalue


def search_minimum_value(data:dict, level:int):
    curr_i = curr_j = 0 # start in position 0, 0
    curr_avg = data[curr_i][curr_j].mean()
    print("%3d, %3d, %.5f" %(curr_i, curr_j, curr_avg))
    while True:
        before_i = curr_i
        before_j = curr_j
        for i in range(-level,level+1):
            pos_i = curr_i + i * 8
            if pos_i < 0 or pos_i > 128:
                continue
            for j in range(-level,level+1):            
                pos_j = curr_j + j * 8
                if pos_j < 0 or pos_j > 128:
                    continue
                if data[pos_i][pos_j].mean() < curr_avg and pvalue(data, pos_i, pos_j, curr_i, curr_j) <= 0.05: 
                    curr_i = pos_i
                    curr_j = pos_j
                    curr_avg = data[pos_i][pos_j].mean()
        if before_i == curr_i and before_j == curr_j:
            break
        print("%3d, %3d, %.5f" %(curr_i, curr_j, curr_avg))
    print(curr_i, curr_j, curr_avg, data[curr_i][curr_j])


def brute_force_minimum_value(data:dict):
    curr_i = curr_j = 0
    curr_avg = data[curr_i][curr_j].mean()
    for pos_i in data.keys():
        for pos_j in data.keys():
            if data[pos_i][pos_j].mean() < curr_avg: 
                curr_i = pos_i
                curr_j = pos_j
                curr_avg = data[pos_i][pos_j].mean()
    print(curr_i, curr_j, curr_avg, data[curr_i][curr_j])


if __name__ == "__main__":

    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
    else:
        print("Example: python3 print_graphic.py")
        exit(0)
    
    data = processing_input(csv_file)
    level = 1

    print("search minimum neighborhod:")
    search_minimum_value(data, level)

    print("Brute force minimum value: ")
    brute_force_minimum_value(data)
