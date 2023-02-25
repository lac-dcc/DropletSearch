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
        try:
            if len(c[2]) > 1:
                for l in c[2][1:]:
                    print(l, end=",")
        except:
            print(c[2], end=",")
    print()

if __name__ == "__main__":

    model = sys.argv[1]
    tuner = sys.argv[2]
    arch = sys.argv[3]

    log_file = "results/%s/%s/%s/cpu.log" % (model, arch, tuner)

    print("%s %s %s" %(model, tuner, arch))

    if model == "resnet-18":
        for i in range(12):
            collect_data(log_file + "_layer_"+str(i)+".log")
    elif model == "vgg-16":
        for i in range(9):
            collect_data(log_file + "_layer_"+str(i)+".log")
    else:
        print("not defined yet")