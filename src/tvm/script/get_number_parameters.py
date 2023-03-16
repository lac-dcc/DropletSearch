import numpy as np
import sys

def get_number_parameters(log):
    import json

    param = []
    f = open(log, "r")
    for line in f.readlines():
        data = json.loads(line)
        if "ansor" in log:
            if len(param) < len(data["i"][1][1]):
                param = data["i"][1][1]
        else:
            for ent in data["config"]["entity"]:
                if ent[0] not in param:
                    param.append(ent[0])
    f.close()

    print(len(param), param)


if __name__ == "__main__":

    log = sys.argv[1]
    get_number_parameters(log)