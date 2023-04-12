import numpy as np
import json
import sys

def process_each_time(log):

    f = open(log, "r")
    c = False
    for l in f.readlines():
        l = l.strip()
        if "Time partial tuning:" in l:
            v = l.split(":")[1].replace(" ","").split(",")
            print("%s,%s,%s,%s" %(v[0],v[1],v[2],v[3]))
        elif "mean (ms)" in l:
            c = True
        elif c:
            l = l.split(" ")
            #print("%s,%s,%.2f"%(l[0],l[-1],v_sum))
            c = False
    f.close() 

def process_time_ansor(log):
    layers = {}
    f = open(log, "r")
    c = 0
    for line in f.readlines():
        data = json.loads(line)
        if "ansor" in log:
            key = str(data["i"][0])
            if key not in layers.keys():
                layers[key] = [c,np.average(data["r"][0])]
                c += 1
            else:
                layers[key].append(np.average(data["r"][0]))
            #layers[data["i"][0]]
    for l in layers:
        #print(layers[l])
        print("%2d, %.4f" %(layers[l][0], 100*min(layers[l][1:])))

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
            v_sum = float(l.replace(",","").split(" ")[2])
        elif "mean (ms)" in l:
            c = True
        elif c:
            l = l.split(" ")
            print("%s,%s,%.2f"%(l[0],l[-1],v_sum))
            c = False
    f.close() 

def p_value(log):

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
            v_sum = float(l.replace(",","").split(" ")[2])
        elif "mean (ms)" in l:
            c = True
        elif c:
            l = l.split(" ")
            print("%s,%s,%.2f"%(l[0],l[-1],v_sum))
            c = False
            break
    f.close() 

if __name__ == "__main__":

    log = sys.argv[1]
    #print(log)
    p_value(log)
