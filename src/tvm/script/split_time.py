import numpy as np
import json
import sys

def process_time(log, t):

    f = open(log, "r")
    c = False
    r = []
    for l in f.readlines():
        l = l.strip()
        if "mean (ms)" in l:
            c = True
        elif "--" in l:
            tmp = [l.split(" ")[1]]
        elif c:
            l = l.split(" ")
            tmp.append(l[0])
            c = False
            r.append(tmp)
    
    print(t, end=",")
    for i in range(len(r)):
        print(r[i][1], end=",")
    print()

    f.close() 

if __name__ == "__main__":

    log = sys.argv[1]
    t = sys.argv[2]
    #print(log)
    process_time(log, t)
