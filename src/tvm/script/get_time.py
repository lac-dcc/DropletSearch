import numpy as np
import json
import sys

def process_time(log):

    f = open(log, "r")
    c = False
    v_sum = 0
    for l in f.readlines():
        l = l.strip()
        if "Time partial tuning:" in l:
            v = l.split(":")[1].replace(" ","").split(",")
            #print("%s,%s,%s" %(v[0],v[1],v[2]))
        elif "Time total tuning:" in l:
            v_sum = float(l.split(":")[1].replace(" ",""))
        elif "Time search" in l:
            v_sum = float(l.split(" ")[2])
        elif "mean (ms)" in l:
            c = True
        elif c:
            l = l.split(" ")
            print("%s,%s,%.2f"%(l[0],l[-1],v_sum))
            c = False
    f.close() 

if __name__ == "__main__":

    log = sys.argv[1]
    #print(log)
    process_time(log)
