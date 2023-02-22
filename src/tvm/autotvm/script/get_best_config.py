import numpy as np
import json
import sys

def get_best_time(log, ms=True):
    f = open(log, "r")
    best_avg, best_std, config = 9999.0, 0.0, ""

    for line in f.readlines():
        data = json.loads(line)
        r = np.mean(data["result"][0])
        if (best_avg > r):
            best_avg = r
            best_std = np.std(data["result"][0])
            config = data
    f.close()

    if ms: # convet to ms
        best_avg *= 1000
        best_std *= 1000
    return best_avg, best_std, config

def collect_data(log):

    best_avg, best_std, config = get_best_time(log)

    print("%.4f, %.4f" %(best_avg, best_std), end=", ")
    
    for c in config["config"]["entity"]:
        print(c[2][1], end=",") if c[0] == "tile_ic" else '-'
        print(c[2][1], end=",") if c[0] == "tile_oc" else '-'
        print(c[2][1], end=",") if c[0] == "tile_ow" else '-'
        print(c[2], end=",") if c[0] == "tile_oh" else '-'
        print(c[2], end=",") if c[0] == "unroll_kw" else '-'
    print()

tuner = ["droplet", "gridsearch", "random", "ga", "xgb"]

for t in tuner:
    print(t)
    for i in range(12):
        collect_data("results/resnet-18/nitro5/"+t+"/cpu_resnet-18_"+t+".log_layer_"+str(i)+".log")