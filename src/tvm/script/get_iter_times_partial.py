import numpy as np
import json
import sys
import glob

def get_best_time(log, ms=True):
    import json

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

if __name__ == "__main__":

    path = sys.argv[1]

    c = 0
    for i in range(100):
        try:
            f = open(path + "/cpu.log_layer_" + str(i) + ".log", "r")
            for l in f.readlines():
                c += 1
            f.close()
        except:
            break
    print("%d" %(c))
