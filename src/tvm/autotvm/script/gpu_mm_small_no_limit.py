import logging
import sys
from itertools import combinations, permutations

import numpy as np
import tvm
import time
import os

from tvm import autotvm, te, testing

@autotvm.template("template_matmul")
def matmul(N, L, M, search_space_tile, dtype="float32"):
    A = te.placeholder((N, L), name="A", dtype=dtype)
    B = te.placeholder((L, M), name="B", dtype=dtype)

    k = te.reduce_axis((0, L), name="k")
    C = te.compute((N, M), lambda i, j: te.sum(A[i, k] * B[k, j], axis=k), name="C")
    s = te.create_schedule(C.op)

    # schedule
    y, x = s[C].op.axis
    k = s[C].op.reduce_axis[0]

    # get the config object
    cfg = autotvm.get_config()

    # define search space
    cfg.define_knob("tile_x", search_space_tile)
    cfg.define_knob("tile_y", search_space_tile)
    cfg.define_knob("tile_z", search_space_tile)    

    # schedule according to config
    x0, x1 = s[C].split(x, cfg["tile_x"].val)
    y0, y1 = s[C].split(y, cfg["tile_y"].val)
    k0, k1 = s[C].split(k, cfg["tile_z"].val)

    # Bind GPU thread indices
    s[C].bind(x0, te.thread_axis("blockIdx.x"))
    s[C].bind(y0, te.thread_axis("blockIdx.y"))
    s[C].bind(k0, te.thread_axis("blockIdx.z"))
    s[C].bind(x1, te.thread_axis("threadIdx.x"))
    s[C].bind(y1, te.thread_axis("threadIdx.y"))
    s[C].bind(k1, te.thread_axis("threadIdx.z"))

    cfg.define_knob("order", [i for i in range(0, 719)])

    if cfg["order"].val == 0:
        s[C].reorder(x0, x1, y0, y1, k0, k1)
    elif cfg["order"].val == 1:
        s[C].reorder(x0, x1, y0, y1, k1, k0)
    elif cfg["order"].val == 2:
        s[C].reorder(x0, x1, y0, k0, y1, k1)
    elif cfg["order"].val == 3:
        s[C].reorder(x0, x1, y0, k0, k1, y1)
    elif cfg["order"].val == 4:
        s[C].reorder(x0, x1, y0, k1, y1, k0)
    elif cfg["order"].val == 5:
        s[C].reorder(x0, x1, y0, k1, k0, y1)
    elif cfg["order"].val == 6:
        s[C].reorder(x0, x1, y1, y0, k0, k1)
    elif cfg["order"].val == 7:
        s[C].reorder(x0, x1, y1, y0, k1, k0)
    elif cfg["order"].val == 8:
        s[C].reorder(x0, x1, y1, k0, y0, k1)
    elif cfg["order"].val == 9:
        s[C].reorder(x0, x1, y1, k0, k1, y0)
    elif cfg["order"].val == 10:
        s[C].reorder(x0, x1, y1, k1, y0, k0)
    elif cfg["order"].val == 11:
        s[C].reorder(x0, x1, y1, k1, k0, y0)
    elif cfg["order"].val == 12:
        s[C].reorder(x0, x1, k0, y0, y1, k1)
    elif cfg["order"].val == 13:
        s[C].reorder(x0, x1, k0, y0, k1, y1)
    elif cfg["order"].val == 14:
        s[C].reorder(x0, x1, k0, y1, y0, k1)
    elif cfg["order"].val == 15:
        s[C].reorder(x0, x1, k0, y1, k1, y0)
    elif cfg["order"].val == 16:
        s[C].reorder(x0, x1, k0, k1, y0, y1)
    elif cfg["order"].val == 17:
        s[C].reorder(x0, x1, k0, k1, y1, y0)
    elif cfg["order"].val == 18:
        s[C].reorder(x0, x1, k1, y0, y1, k0)
    elif cfg["order"].val == 19:
        s[C].reorder(x0, x1, k1, y0, k0, y1)
    elif cfg["order"].val == 20:
        s[C].reorder(x0, x1, k1, y1, y0, k0)
    elif cfg["order"].val == 21:
        s[C].reorder(x0, x1, k1, y1, k0, y0)
    elif cfg["order"].val == 22:
        s[C].reorder(x0, x1, k1, k0, y0, y1)
    elif cfg["order"].val == 23:
        s[C].reorder(x0, x1, k1, k0, y1, y0)
    elif cfg["order"].val == 24:
        s[C].reorder(x0, y0, x1, y1, k0, k1)
    elif cfg["order"].val == 25:
        s[C].reorder(x0, y0, x1, y1, k1, k0)
    elif cfg["order"].val == 26:
        s[C].reorder(x0, y0, x1, k0, y1, k1)
    elif cfg["order"].val == 27:
        s[C].reorder(x0, y0, x1, k0, k1, y1)
    elif cfg["order"].val == 28:
        s[C].reorder(x0, y0, x1, k1, y1, k0)
    elif cfg["order"].val == 29:
        s[C].reorder(x0, y0, x1, k1, k0, y1)
    elif cfg["order"].val == 30:
        s[C].reorder(x0, y0, y1, x1, k0, k1)
    elif cfg["order"].val == 31:
        s[C].reorder(x0, y0, y1, x1, k1, k0)
    elif cfg["order"].val == 32:
        s[C].reorder(x0, y0, y1, k0, x1, k1)
    elif cfg["order"].val == 33:
        s[C].reorder(x0, y0, y1, k0, k1, x1)
    elif cfg["order"].val == 34:
        s[C].reorder(x0, y0, y1, k1, x1, k0)
    elif cfg["order"].val == 35:
        s[C].reorder(x0, y0, y1, k1, k0, x1)
    elif cfg["order"].val == 36:
        s[C].reorder(x0, y0, k0, x1, y1, k1)
    elif cfg["order"].val == 37:
        s[C].reorder(x0, y0, k0, x1, k1, y1)
    elif cfg["order"].val == 38:
        s[C].reorder(x0, y0, k0, y1, x1, k1)
    elif cfg["order"].val == 39:
        s[C].reorder(x0, y0, k0, y1, k1, x1)
    elif cfg["order"].val == 40:
        s[C].reorder(x0, y0, k0, k1, x1, y1)
    elif cfg["order"].val == 41:
        s[C].reorder(x0, y0, k0, k1, y1, x1)
    elif cfg["order"].val == 42:
        s[C].reorder(x0, y0, k1, x1, y1, k0)
    elif cfg["order"].val == 43:
        s[C].reorder(x0, y0, k1, x1, k0, y1)
    elif cfg["order"].val == 44:
        s[C].reorder(x0, y0, k1, y1, x1, k0)
    elif cfg["order"].val == 45:
        s[C].reorder(x0, y0, k1, y1, k0, x1)
    elif cfg["order"].val == 46:
        s[C].reorder(x0, y0, k1, k0, x1, y1)
    elif cfg["order"].val == 47:
        s[C].reorder(x0, y0, k1, k0, y1, x1)
    elif cfg["order"].val == 48:
        s[C].reorder(x0, y1, x1, y0, k0, k1)
    elif cfg["order"].val == 49:
        s[C].reorder(x0, y1, x1, y0, k1, k0)
    elif cfg["order"].val == 50:
        s[C].reorder(x0, y1, x1, k0, y0, k1)
    elif cfg["order"].val == 51:
        s[C].reorder(x0, y1, x1, k0, k1, y0)
    elif cfg["order"].val == 52:
        s[C].reorder(x0, y1, x1, k1, y0, k0)
    elif cfg["order"].val == 53:
        s[C].reorder(x0, y1, x1, k1, k0, y0)
    elif cfg["order"].val == 54:
        s[C].reorder(x0, y1, y0, x1, k0, k1)
    elif cfg["order"].val == 55:
        s[C].reorder(x0, y1, y0, x1, k1, k0)
    elif cfg["order"].val == 56:
        s[C].reorder(x0, y1, y0, k0, x1, k1)
    elif cfg["order"].val == 57:
        s[C].reorder(x0, y1, y0, k0, k1, x1)
    elif cfg["order"].val == 58:
        s[C].reorder(x0, y1, y0, k1, x1, k0)
    elif cfg["order"].val == 59:
        s[C].reorder(x0, y1, y0, k1, k0, x1)
    elif cfg["order"].val == 60:
        s[C].reorder(x0, y1, k0, x1, y0, k1)
    elif cfg["order"].val == 61:
        s[C].reorder(x0, y1, k0, x1, k1, y0)
    elif cfg["order"].val == 62:
        s[C].reorder(x0, y1, k0, y0, x1, k1)
    elif cfg["order"].val == 63:
        s[C].reorder(x0, y1, k0, y0, k1, x1)
    elif cfg["order"].val == 64:
        s[C].reorder(x0, y1, k0, k1, x1, y0)
    elif cfg["order"].val == 65:
        s[C].reorder(x0, y1, k0, k1, y0, x1)
    elif cfg["order"].val == 66:
        s[C].reorder(x0, y1, k1, x1, y0, k0)
    elif cfg["order"].val == 67:
        s[C].reorder(x0, y1, k1, x1, k0, y0)
    elif cfg["order"].val == 68:
        s[C].reorder(x0, y1, k1, y0, x1, k0)
    elif cfg["order"].val == 69:
        s[C].reorder(x0, y1, k1, y0, k0, x1)
    elif cfg["order"].val == 70:
        s[C].reorder(x0, y1, k1, k0, x1, y0)
    elif cfg["order"].val == 71:
        s[C].reorder(x0, y1, k1, k0, y0, x1)
    elif cfg["order"].val == 72:
        s[C].reorder(x0, k0, x1, y0, y1, k1)
    elif cfg["order"].val == 73:
        s[C].reorder(x0, k0, x1, y0, k1, y1)
    elif cfg["order"].val == 74:
        s[C].reorder(x0, k0, x1, y1, y0, k1)
    elif cfg["order"].val == 75:
        s[C].reorder(x0, k0, x1, y1, k1, y0)
    elif cfg["order"].val == 76:
        s[C].reorder(x0, k0, x1, k1, y0, y1)
    elif cfg["order"].val == 77:
        s[C].reorder(x0, k0, x1, k1, y1, y0)
    elif cfg["order"].val == 78:
        s[C].reorder(x0, k0, y0, x1, y1, k1)
    elif cfg["order"].val == 79:
        s[C].reorder(x0, k0, y0, x1, k1, y1)
    elif cfg["order"].val == 80:
        s[C].reorder(x0, k0, y0, y1, x1, k1)
    elif cfg["order"].val == 81:
        s[C].reorder(x0, k0, y0, y1, k1, x1)
    elif cfg["order"].val == 82:
        s[C].reorder(x0, k0, y0, k1, x1, y1)
    elif cfg["order"].val == 83:
        s[C].reorder(x0, k0, y0, k1, y1, x1)
    elif cfg["order"].val == 84:
        s[C].reorder(x0, k0, y1, x1, y0, k1)
    elif cfg["order"].val == 85:
        s[C].reorder(x0, k0, y1, x1, k1, y0)
    elif cfg["order"].val == 86:
        s[C].reorder(x0, k0, y1, y0, x1, k1)
    elif cfg["order"].val == 87:
        s[C].reorder(x0, k0, y1, y0, k1, x1)
    elif cfg["order"].val == 88:
        s[C].reorder(x0, k0, y1, k1, x1, y0)
    elif cfg["order"].val == 89:
        s[C].reorder(x0, k0, y1, k1, y0, x1)
    elif cfg["order"].val == 90:
        s[C].reorder(x0, k0, k1, x1, y0, y1)
    elif cfg["order"].val == 91:
        s[C].reorder(x0, k0, k1, x1, y1, y0)
    elif cfg["order"].val == 92:
        s[C].reorder(x0, k0, k1, y0, x1, y1)
    elif cfg["order"].val == 93:
        s[C].reorder(x0, k0, k1, y0, y1, x1)
    elif cfg["order"].val == 94:
        s[C].reorder(x0, k0, k1, y1, x1, y0)
    elif cfg["order"].val == 95:
        s[C].reorder(x0, k0, k1, y1, y0, x1)
    elif cfg["order"].val == 96:
        s[C].reorder(x0, k1, x1, y0, y1, k0)
    elif cfg["order"].val == 97:
        s[C].reorder(x0, k1, x1, y0, k0, y1)
    elif cfg["order"].val == 98:
        s[C].reorder(x0, k1, x1, y1, y0, k0)
    elif cfg["order"].val == 99:
        s[C].reorder(x0, k1, x1, y1, k0, y0)
    elif cfg["order"].val == 100:
        s[C].reorder(x0, k1, x1, k0, y0, y1)
    elif cfg["order"].val == 101:
        s[C].reorder(x0, k1, x1, k0, y1, y0)
    elif cfg["order"].val == 102:
        s[C].reorder(x0, k1, y0, x1, y1, k0)
    elif cfg["order"].val == 103:
        s[C].reorder(x0, k1, y0, x1, k0, y1)
    elif cfg["order"].val == 104:
        s[C].reorder(x0, k1, y0, y1, x1, k0)
    elif cfg["order"].val == 105:
        s[C].reorder(x0, k1, y0, y1, k0, x1)
    elif cfg["order"].val == 106:
        s[C].reorder(x0, k1, y0, k0, x1, y1)
    elif cfg["order"].val == 107:
        s[C].reorder(x0, k1, y0, k0, y1, x1)
    elif cfg["order"].val == 108:
        s[C].reorder(x0, k1, y1, x1, y0, k0)
    elif cfg["order"].val == 109:
        s[C].reorder(x0, k1, y1, x1, k0, y0)
    elif cfg["order"].val == 110:
        s[C].reorder(x0, k1, y1, y0, x1, k0)
    elif cfg["order"].val == 111:
        s[C].reorder(x0, k1, y1, y0, k0, x1)
    elif cfg["order"].val == 112:
        s[C].reorder(x0, k1, y1, k0, x1, y0)
    elif cfg["order"].val == 113:
        s[C].reorder(x0, k1, y1, k0, y0, x1)
    elif cfg["order"].val == 114:
        s[C].reorder(x0, k1, k0, x1, y0, y1)
    elif cfg["order"].val == 115:
        s[C].reorder(x0, k1, k0, x1, y1, y0)
    elif cfg["order"].val == 116:
        s[C].reorder(x0, k1, k0, y0, x1, y1)
    elif cfg["order"].val == 117:
        s[C].reorder(x0, k1, k0, y0, y1, x1)
    elif cfg["order"].val == 118:
        s[C].reorder(x0, k1, k0, y1, x1, y0)
    elif cfg["order"].val == 119:
        s[C].reorder(x0, k1, k0, y1, y0, x1)
    elif cfg["order"].val == 120:
        s[C].reorder(x1, x0, y0, y1, k0, k1)
    elif cfg["order"].val == 121:
        s[C].reorder(x1, x0, y0, y1, k1, k0)
    elif cfg["order"].val == 122:
        s[C].reorder(x1, x0, y0, k0, y1, k1)
    elif cfg["order"].val == 123:
        s[C].reorder(x1, x0, y0, k0, k1, y1)
    elif cfg["order"].val == 124:
        s[C].reorder(x1, x0, y0, k1, y1, k0)
    elif cfg["order"].val == 125:
        s[C].reorder(x1, x0, y0, k1, k0, y1)
    elif cfg["order"].val == 126:
        s[C].reorder(x1, x0, y1, y0, k0, k1)
    elif cfg["order"].val == 127:
        s[C].reorder(x1, x0, y1, y0, k1, k0)
    elif cfg["order"].val == 128:
        s[C].reorder(x1, x0, y1, k0, y0, k1)
    elif cfg["order"].val == 129:
        s[C].reorder(x1, x0, y1, k0, k1, y0)
    elif cfg["order"].val == 130:
        s[C].reorder(x1, x0, y1, k1, y0, k0)
    elif cfg["order"].val == 131:
        s[C].reorder(x1, x0, y1, k1, k0, y0)
    elif cfg["order"].val == 132:
        s[C].reorder(x1, x0, k0, y0, y1, k1)
    elif cfg["order"].val == 133:
        s[C].reorder(x1, x0, k0, y0, k1, y1)
    elif cfg["order"].val == 134:
        s[C].reorder(x1, x0, k0, y1, y0, k1)
    elif cfg["order"].val == 135:
        s[C].reorder(x1, x0, k0, y1, k1, y0)
    elif cfg["order"].val == 136:
        s[C].reorder(x1, x0, k0, k1, y0, y1)
    elif cfg["order"].val == 137:
        s[C].reorder(x1, x0, k0, k1, y1, y0)
    elif cfg["order"].val == 138:
        s[C].reorder(x1, x0, k1, y0, y1, k0)
    elif cfg["order"].val == 139:
        s[C].reorder(x1, x0, k1, y0, k0, y1)
    elif cfg["order"].val == 140:
        s[C].reorder(x1, x0, k1, y1, y0, k0)
    elif cfg["order"].val == 141:
        s[C].reorder(x1, x0, k1, y1, k0, y0)
    elif cfg["order"].val == 142:
        s[C].reorder(x1, x0, k1, k0, y0, y1)
    elif cfg["order"].val == 143:
        s[C].reorder(x1, x0, k1, k0, y1, y0)
    elif cfg["order"].val == 144:
        s[C].reorder(x1, y0, x0, y1, k0, k1)
    elif cfg["order"].val == 145:
        s[C].reorder(x1, y0, x0, y1, k1, k0)
    elif cfg["order"].val == 146:
        s[C].reorder(x1, y0, x0, k0, y1, k1)
    elif cfg["order"].val == 147:
        s[C].reorder(x1, y0, x0, k0, k1, y1)
    elif cfg["order"].val == 148:
        s[C].reorder(x1, y0, x0, k1, y1, k0)
    elif cfg["order"].val == 149:
        s[C].reorder(x1, y0, x0, k1, k0, y1)
    elif cfg["order"].val == 150:
        s[C].reorder(x1, y0, y1, x0, k0, k1)
    elif cfg["order"].val == 151:
        s[C].reorder(x1, y0, y1, x0, k1, k0)
    elif cfg["order"].val == 152:
        s[C].reorder(x1, y0, y1, k0, x0, k1)
    elif cfg["order"].val == 153:
        s[C].reorder(x1, y0, y1, k0, k1, x0)
    elif cfg["order"].val == 154:
        s[C].reorder(x1, y0, y1, k1, x0, k0)
    elif cfg["order"].val == 155:
        s[C].reorder(x1, y0, y1, k1, k0, x0)
    elif cfg["order"].val == 156:
        s[C].reorder(x1, y0, k0, x0, y1, k1)
    elif cfg["order"].val == 157:
        s[C].reorder(x1, y0, k0, x0, k1, y1)
    elif cfg["order"].val == 158:
        s[C].reorder(x1, y0, k0, y1, x0, k1)
    elif cfg["order"].val == 159:
        s[C].reorder(x1, y0, k0, y1, k1, x0)
    elif cfg["order"].val == 160:
        s[C].reorder(x1, y0, k0, k1, x0, y1)
    elif cfg["order"].val == 161:
        s[C].reorder(x1, y0, k0, k1, y1, x0)
    elif cfg["order"].val == 162:
        s[C].reorder(x1, y0, k1, x0, y1, k0)
    elif cfg["order"].val == 163:
        s[C].reorder(x1, y0, k1, x0, k0, y1)
    elif cfg["order"].val == 164:
        s[C].reorder(x1, y0, k1, y1, x0, k0)
    elif cfg["order"].val == 165:
        s[C].reorder(x1, y0, k1, y1, k0, x0)
    elif cfg["order"].val == 166:
        s[C].reorder(x1, y0, k1, k0, x0, y1)
    elif cfg["order"].val == 167:
        s[C].reorder(x1, y0, k1, k0, y1, x0)
    elif cfg["order"].val == 168:
        s[C].reorder(x1, y1, x0, y0, k0, k1)
    elif cfg["order"].val == 169:
        s[C].reorder(x1, y1, x0, y0, k1, k0)
    elif cfg["order"].val == 170:
        s[C].reorder(x1, y1, x0, k0, y0, k1)
    elif cfg["order"].val == 171:
        s[C].reorder(x1, y1, x0, k0, k1, y0)
    elif cfg["order"].val == 172:
        s[C].reorder(x1, y1, x0, k1, y0, k0)
    elif cfg["order"].val == 173:
        s[C].reorder(x1, y1, x0, k1, k0, y0)
    elif cfg["order"].val == 174:
        s[C].reorder(x1, y1, y0, x0, k0, k1)
    elif cfg["order"].val == 175:
        s[C].reorder(x1, y1, y0, x0, k1, k0)
    elif cfg["order"].val == 176:
        s[C].reorder(x1, y1, y0, k0, x0, k1)
    elif cfg["order"].val == 177:
        s[C].reorder(x1, y1, y0, k0, k1, x0)
    elif cfg["order"].val == 178:
        s[C].reorder(x1, y1, y0, k1, x0, k0)
    elif cfg["order"].val == 179:
        s[C].reorder(x1, y1, y0, k1, k0, x0)
    elif cfg["order"].val == 180:
        s[C].reorder(x1, y1, k0, x0, y0, k1)
    elif cfg["order"].val == 181:
        s[C].reorder(x1, y1, k0, x0, k1, y0)
    elif cfg["order"].val == 182:
        s[C].reorder(x1, y1, k0, y0, x0, k1)
    elif cfg["order"].val == 183:
        s[C].reorder(x1, y1, k0, y0, k1, x0)
    elif cfg["order"].val == 184:
        s[C].reorder(x1, y1, k0, k1, x0, y0)
    elif cfg["order"].val == 185:
        s[C].reorder(x1, y1, k0, k1, y0, x0)
    elif cfg["order"].val == 186:
        s[C].reorder(x1, y1, k1, x0, y0, k0)
    elif cfg["order"].val == 187:
        s[C].reorder(x1, y1, k1, x0, k0, y0)
    elif cfg["order"].val == 188:
        s[C].reorder(x1, y1, k1, y0, x0, k0)
    elif cfg["order"].val == 189:
        s[C].reorder(x1, y1, k1, y0, k0, x0)
    elif cfg["order"].val == 190:
        s[C].reorder(x1, y1, k1, k0, x0, y0)
    elif cfg["order"].val == 191:
        s[C].reorder(x1, y1, k1, k0, y0, x0)
    elif cfg["order"].val == 192:
        s[C].reorder(x1, k0, x0, y0, y1, k1)
    elif cfg["order"].val == 193:
        s[C].reorder(x1, k0, x0, y0, k1, y1)
    elif cfg["order"].val == 194:
        s[C].reorder(x1, k0, x0, y1, y0, k1)
    elif cfg["order"].val == 195:
        s[C].reorder(x1, k0, x0, y1, k1, y0)
    elif cfg["order"].val == 196:
        s[C].reorder(x1, k0, x0, k1, y0, y1)
    elif cfg["order"].val == 197:
        s[C].reorder(x1, k0, x0, k1, y1, y0)
    elif cfg["order"].val == 198:
        s[C].reorder(x1, k0, y0, x0, y1, k1)
    elif cfg["order"].val == 199:
        s[C].reorder(x1, k0, y0, x0, k1, y1)
    elif cfg["order"].val == 200:
        s[C].reorder(x1, k0, y0, y1, x0, k1)
    elif cfg["order"].val == 201:
        s[C].reorder(x1, k0, y0, y1, k1, x0)
    elif cfg["order"].val == 202:
        s[C].reorder(x1, k0, y0, k1, x0, y1)
    elif cfg["order"].val == 203:
        s[C].reorder(x1, k0, y0, k1, y1, x0)
    elif cfg["order"].val == 204:
        s[C].reorder(x1, k0, y1, x0, y0, k1)
    elif cfg["order"].val == 205:
        s[C].reorder(x1, k0, y1, x0, k1, y0)
    elif cfg["order"].val == 206:
        s[C].reorder(x1, k0, y1, y0, x0, k1)
    elif cfg["order"].val == 207:
        s[C].reorder(x1, k0, y1, y0, k1, x0)
    elif cfg["order"].val == 208:
        s[C].reorder(x1, k0, y1, k1, x0, y0)
    elif cfg["order"].val == 209:
        s[C].reorder(x1, k0, y1, k1, y0, x0)
    elif cfg["order"].val == 210:
        s[C].reorder(x1, k0, k1, x0, y0, y1)
    elif cfg["order"].val == 211:
        s[C].reorder(x1, k0, k1, x0, y1, y0)
    elif cfg["order"].val == 212:
        s[C].reorder(x1, k0, k1, y0, x0, y1)
    elif cfg["order"].val == 213:
        s[C].reorder(x1, k0, k1, y0, y1, x0)
    elif cfg["order"].val == 214:
        s[C].reorder(x1, k0, k1, y1, x0, y0)
    elif cfg["order"].val == 215:
        s[C].reorder(x1, k0, k1, y1, y0, x0)
    elif cfg["order"].val == 216:
        s[C].reorder(x1, k1, x0, y0, y1, k0)
    elif cfg["order"].val == 217:
        s[C].reorder(x1, k1, x0, y0, k0, y1)
    elif cfg["order"].val == 218:
        s[C].reorder(x1, k1, x0, y1, y0, k0)
    elif cfg["order"].val == 219:
        s[C].reorder(x1, k1, x0, y1, k0, y0)
    elif cfg["order"].val == 220:
        s[C].reorder(x1, k1, x0, k0, y0, y1)
    elif cfg["order"].val == 221:
        s[C].reorder(x1, k1, x0, k0, y1, y0)
    elif cfg["order"].val == 222:
        s[C].reorder(x1, k1, y0, x0, y1, k0)
    elif cfg["order"].val == 223:
        s[C].reorder(x1, k1, y0, x0, k0, y1)
    elif cfg["order"].val == 224:
        s[C].reorder(x1, k1, y0, y1, x0, k0)
    elif cfg["order"].val == 225:
        s[C].reorder(x1, k1, y0, y1, k0, x0)
    elif cfg["order"].val == 226:
        s[C].reorder(x1, k1, y0, k0, x0, y1)
    elif cfg["order"].val == 227:
        s[C].reorder(x1, k1, y0, k0, y1, x0)
    elif cfg["order"].val == 228:
        s[C].reorder(x1, k1, y1, x0, y0, k0)
    elif cfg["order"].val == 229:
        s[C].reorder(x1, k1, y1, x0, k0, y0)
    elif cfg["order"].val == 230:
        s[C].reorder(x1, k1, y1, y0, x0, k0)
    elif cfg["order"].val == 231:
        s[C].reorder(x1, k1, y1, y0, k0, x0)
    elif cfg["order"].val == 232:
        s[C].reorder(x1, k1, y1, k0, x0, y0)
    elif cfg["order"].val == 233:
        s[C].reorder(x1, k1, y1, k0, y0, x0)
    elif cfg["order"].val == 234:
        s[C].reorder(x1, k1, k0, x0, y0, y1)
    elif cfg["order"].val == 235:
        s[C].reorder(x1, k1, k0, x0, y1, y0)
    elif cfg["order"].val == 236:
        s[C].reorder(x1, k1, k0, y0, x0, y1)
    elif cfg["order"].val == 237:
        s[C].reorder(x1, k1, k0, y0, y1, x0)
    elif cfg["order"].val == 238:
        s[C].reorder(x1, k1, k0, y1, x0, y0)
    elif cfg["order"].val == 239:
        s[C].reorder(x1, k1, k0, y1, y0, x0)
    elif cfg["order"].val == 240:
        s[C].reorder(y0, x0, x1, y1, k0, k1)
    elif cfg["order"].val == 241:
        s[C].reorder(y0, x0, x1, y1, k1, k0)
    elif cfg["order"].val == 242:
        s[C].reorder(y0, x0, x1, k0, y1, k1)
    elif cfg["order"].val == 243:
        s[C].reorder(y0, x0, x1, k0, k1, y1)
    elif cfg["order"].val == 244:
        s[C].reorder(y0, x0, x1, k1, y1, k0)
    elif cfg["order"].val == 245:
        s[C].reorder(y0, x0, x1, k1, k0, y1)
    elif cfg["order"].val == 246:
        s[C].reorder(y0, x0, y1, x1, k0, k1)
    elif cfg["order"].val == 247:
        s[C].reorder(y0, x0, y1, x1, k1, k0)
    elif cfg["order"].val == 248:
        s[C].reorder(y0, x0, y1, k0, x1, k1)
    elif cfg["order"].val == 249:
        s[C].reorder(y0, x0, y1, k0, k1, x1)
    elif cfg["order"].val == 250:
        s[C].reorder(y0, x0, y1, k1, x1, k0)
    elif cfg["order"].val == 251:
        s[C].reorder(y0, x0, y1, k1, k0, x1)
    elif cfg["order"].val == 252:
        s[C].reorder(y0, x0, k0, x1, y1, k1)
    elif cfg["order"].val == 253:
        s[C].reorder(y0, x0, k0, x1, k1, y1)
    elif cfg["order"].val == 254:
        s[C].reorder(y0, x0, k0, y1, x1, k1)
    elif cfg["order"].val == 255:
        s[C].reorder(y0, x0, k0, y1, k1, x1)
    elif cfg["order"].val == 256:
        s[C].reorder(y0, x0, k0, k1, x1, y1)
    elif cfg["order"].val == 257:
        s[C].reorder(y0, x0, k0, k1, y1, x1)
    elif cfg["order"].val == 258:
        s[C].reorder(y0, x0, k1, x1, y1, k0)
    elif cfg["order"].val == 259:
        s[C].reorder(y0, x0, k1, x1, k0, y1)
    elif cfg["order"].val == 260:
        s[C].reorder(y0, x0, k1, y1, x1, k0)
    elif cfg["order"].val == 261:
        s[C].reorder(y0, x0, k1, y1, k0, x1)
    elif cfg["order"].val == 262:
        s[C].reorder(y0, x0, k1, k0, x1, y1)
    elif cfg["order"].val == 263:
        s[C].reorder(y0, x0, k1, k0, y1, x1)
    elif cfg["order"].val == 264:
        s[C].reorder(y0, x1, x0, y1, k0, k1)
    elif cfg["order"].val == 265:
        s[C].reorder(y0, x1, x0, y1, k1, k0)
    elif cfg["order"].val == 266:
        s[C].reorder(y0, x1, x0, k0, y1, k1)
    elif cfg["order"].val == 267:
        s[C].reorder(y0, x1, x0, k0, k1, y1)
    elif cfg["order"].val == 268:
        s[C].reorder(y0, x1, x0, k1, y1, k0)
    elif cfg["order"].val == 269:
        s[C].reorder(y0, x1, x0, k1, k0, y1)
    elif cfg["order"].val == 270:
        s[C].reorder(y0, x1, y1, x0, k0, k1)
    elif cfg["order"].val == 271:
        s[C].reorder(y0, x1, y1, x0, k1, k0)
    elif cfg["order"].val == 272:
        s[C].reorder(y0, x1, y1, k0, x0, k1)
    elif cfg["order"].val == 273:
        s[C].reorder(y0, x1, y1, k0, k1, x0)
    elif cfg["order"].val == 274:
        s[C].reorder(y0, x1, y1, k1, x0, k0)
    elif cfg["order"].val == 275:
        s[C].reorder(y0, x1, y1, k1, k0, x0)
    elif cfg["order"].val == 276:
        s[C].reorder(y0, x1, k0, x0, y1, k1)
    elif cfg["order"].val == 277:
        s[C].reorder(y0, x1, k0, x0, k1, y1)
    elif cfg["order"].val == 278:
        s[C].reorder(y0, x1, k0, y1, x0, k1)
    elif cfg["order"].val == 279:
        s[C].reorder(y0, x1, k0, y1, k1, x0)
    elif cfg["order"].val == 280:
        s[C].reorder(y0, x1, k0, k1, x0, y1)
    elif cfg["order"].val == 281:
        s[C].reorder(y0, x1, k0, k1, y1, x0)
    elif cfg["order"].val == 282:
        s[C].reorder(y0, x1, k1, x0, y1, k0)
    elif cfg["order"].val == 283:
        s[C].reorder(y0, x1, k1, x0, k0, y1)
    elif cfg["order"].val == 284:
        s[C].reorder(y0, x1, k1, y1, x0, k0)
    elif cfg["order"].val == 285:
        s[C].reorder(y0, x1, k1, y1, k0, x0)
    elif cfg["order"].val == 286:
        s[C].reorder(y0, x1, k1, k0, x0, y1)
    elif cfg["order"].val == 287:
        s[C].reorder(y0, x1, k1, k0, y1, x0)
    elif cfg["order"].val == 288:
        s[C].reorder(y0, y1, x0, x1, k0, k1)
    elif cfg["order"].val == 289:
        s[C].reorder(y0, y1, x0, x1, k1, k0)
    elif cfg["order"].val == 290:
        s[C].reorder(y0, y1, x0, k0, x1, k1)
    elif cfg["order"].val == 291:
        s[C].reorder(y0, y1, x0, k0, k1, x1)
    elif cfg["order"].val == 292:
        s[C].reorder(y0, y1, x0, k1, x1, k0)
    elif cfg["order"].val == 293:
        s[C].reorder(y0, y1, x0, k1, k0, x1)
    elif cfg["order"].val == 294:
        s[C].reorder(y0, y1, x1, x0, k0, k1)
    elif cfg["order"].val == 295:
        s[C].reorder(y0, y1, x1, x0, k1, k0)
    elif cfg["order"].val == 296:
        s[C].reorder(y0, y1, x1, k0, x0, k1)
    elif cfg["order"].val == 297:
        s[C].reorder(y0, y1, x1, k0, k1, x0)
    elif cfg["order"].val == 298:
        s[C].reorder(y0, y1, x1, k1, x0, k0)
    elif cfg["order"].val == 299:
        s[C].reorder(y0, y1, x1, k1, k0, x0)
    elif cfg["order"].val == 300:
        s[C].reorder(y0, y1, k0, x0, x1, k1)
    elif cfg["order"].val == 301:
        s[C].reorder(y0, y1, k0, x0, k1, x1)
    elif cfg["order"].val == 302:
        s[C].reorder(y0, y1, k0, x1, x0, k1)
    elif cfg["order"].val == 303:
        s[C].reorder(y0, y1, k0, x1, k1, x0)
    elif cfg["order"].val == 304:
        s[C].reorder(y0, y1, k0, k1, x0, x1)
    elif cfg["order"].val == 305:
        s[C].reorder(y0, y1, k0, k1, x1, x0)
    elif cfg["order"].val == 306:
        s[C].reorder(y0, y1, k1, x0, x1, k0)
    elif cfg["order"].val == 307:
        s[C].reorder(y0, y1, k1, x0, k0, x1)
    elif cfg["order"].val == 308:
        s[C].reorder(y0, y1, k1, x1, x0, k0)
    elif cfg["order"].val == 309:
        s[C].reorder(y0, y1, k1, x1, k0, x0)
    elif cfg["order"].val == 310:
        s[C].reorder(y0, y1, k1, k0, x0, x1)
    elif cfg["order"].val == 311:
        s[C].reorder(y0, y1, k1, k0, x1, x0)
    elif cfg["order"].val == 312:
        s[C].reorder(y0, k0, x0, x1, y1, k1)
    elif cfg["order"].val == 313:
        s[C].reorder(y0, k0, x0, x1, k1, y1)
    elif cfg["order"].val == 314:
        s[C].reorder(y0, k0, x0, y1, x1, k1)
    elif cfg["order"].val == 315:
        s[C].reorder(y0, k0, x0, y1, k1, x1)
    elif cfg["order"].val == 316:
        s[C].reorder(y0, k0, x0, k1, x1, y1)
    elif cfg["order"].val == 317:
        s[C].reorder(y0, k0, x0, k1, y1, x1)
    elif cfg["order"].val == 318:
        s[C].reorder(y0, k0, x1, x0, y1, k1)
    elif cfg["order"].val == 319:
        s[C].reorder(y0, k0, x1, x0, k1, y1)
    elif cfg["order"].val == 320:
        s[C].reorder(y0, k0, x1, y1, x0, k1)
    elif cfg["order"].val == 321:
        s[C].reorder(y0, k0, x1, y1, k1, x0)
    elif cfg["order"].val == 322:
        s[C].reorder(y0, k0, x1, k1, x0, y1)
    elif cfg["order"].val == 323:
        s[C].reorder(y0, k0, x1, k1, y1, x0)
    elif cfg["order"].val == 324:
        s[C].reorder(y0, k0, y1, x0, x1, k1)
    elif cfg["order"].val == 325:
        s[C].reorder(y0, k0, y1, x0, k1, x1)
    elif cfg["order"].val == 326:
        s[C].reorder(y0, k0, y1, x1, x0, k1)
    elif cfg["order"].val == 327:
        s[C].reorder(y0, k0, y1, x1, k1, x0)
    elif cfg["order"].val == 328:
        s[C].reorder(y0, k0, y1, k1, x0, x1)
    elif cfg["order"].val == 329:
        s[C].reorder(y0, k0, y1, k1, x1, x0)
    elif cfg["order"].val == 330:
        s[C].reorder(y0, k0, k1, x0, x1, y1)
    elif cfg["order"].val == 331:
        s[C].reorder(y0, k0, k1, x0, y1, x1)
    elif cfg["order"].val == 332:
        s[C].reorder(y0, k0, k1, x1, x0, y1)
    elif cfg["order"].val == 333:
        s[C].reorder(y0, k0, k1, x1, y1, x0)
    elif cfg["order"].val == 334:
        s[C].reorder(y0, k0, k1, y1, x0, x1)
    elif cfg["order"].val == 335:
        s[C].reorder(y0, k0, k1, y1, x1, x0)
    elif cfg["order"].val == 336:
        s[C].reorder(y0, k1, x0, x1, y1, k0)
    elif cfg["order"].val == 337:
        s[C].reorder(y0, k1, x0, x1, k0, y1)
    elif cfg["order"].val == 338:
        s[C].reorder(y0, k1, x0, y1, x1, k0)
    elif cfg["order"].val == 339:
        s[C].reorder(y0, k1, x0, y1, k0, x1)
    elif cfg["order"].val == 340:
        s[C].reorder(y0, k1, x0, k0, x1, y1)
    elif cfg["order"].val == 341:
        s[C].reorder(y0, k1, x0, k0, y1, x1)
    elif cfg["order"].val == 342:
        s[C].reorder(y0, k1, x1, x0, y1, k0)
    elif cfg["order"].val == 343:
        s[C].reorder(y0, k1, x1, x0, k0, y1)
    elif cfg["order"].val == 344:
        s[C].reorder(y0, k1, x1, y1, x0, k0)
    elif cfg["order"].val == 345:
        s[C].reorder(y0, k1, x1, y1, k0, x0)
    elif cfg["order"].val == 346:
        s[C].reorder(y0, k1, x1, k0, x0, y1)
    elif cfg["order"].val == 347:
        s[C].reorder(y0, k1, x1, k0, y1, x0)
    elif cfg["order"].val == 348:
        s[C].reorder(y0, k1, y1, x0, x1, k0)
    elif cfg["order"].val == 349:
        s[C].reorder(y0, k1, y1, x0, k0, x1)
    elif cfg["order"].val == 350:
        s[C].reorder(y0, k1, y1, x1, x0, k0)
    elif cfg["order"].val == 351:
        s[C].reorder(y0, k1, y1, x1, k0, x0)
    elif cfg["order"].val == 352:
        s[C].reorder(y0, k1, y1, k0, x0, x1)
    elif cfg["order"].val == 353:
        s[C].reorder(y0, k1, y1, k0, x1, x0)
    elif cfg["order"].val == 354:
        s[C].reorder(y0, k1, k0, x0, x1, y1)
    elif cfg["order"].val == 355:
        s[C].reorder(y0, k1, k0, x0, y1, x1)
    elif cfg["order"].val == 356:
        s[C].reorder(y0, k1, k0, x1, x0, y1)
    elif cfg["order"].val == 357:
        s[C].reorder(y0, k1, k0, x1, y1, x0)
    elif cfg["order"].val == 358:
        s[C].reorder(y0, k1, k0, y1, x0, x1)
    elif cfg["order"].val == 359:
        s[C].reorder(y0, k1, k0, y1, x1, x0)
    elif cfg["order"].val == 360:
        s[C].reorder(y1, x0, x1, y0, k0, k1)
    elif cfg["order"].val == 361:
        s[C].reorder(y1, x0, x1, y0, k1, k0)
    elif cfg["order"].val == 362:
        s[C].reorder(y1, x0, x1, k0, y0, k1)
    elif cfg["order"].val == 363:
        s[C].reorder(y1, x0, x1, k0, k1, y0)
    elif cfg["order"].val == 364:
        s[C].reorder(y1, x0, x1, k1, y0, k0)
    elif cfg["order"].val == 365:
        s[C].reorder(y1, x0, x1, k1, k0, y0)
    elif cfg["order"].val == 366:
        s[C].reorder(y1, x0, y0, x1, k0, k1)
    elif cfg["order"].val == 367:
        s[C].reorder(y1, x0, y0, x1, k1, k0)
    elif cfg["order"].val == 368:
        s[C].reorder(y1, x0, y0, k0, x1, k1)
    elif cfg["order"].val == 369:
        s[C].reorder(y1, x0, y0, k0, k1, x1)
    elif cfg["order"].val == 370:
        s[C].reorder(y1, x0, y0, k1, x1, k0)
    elif cfg["order"].val == 371:
        s[C].reorder(y1, x0, y0, k1, k0, x1)
    elif cfg["order"].val == 372:
        s[C].reorder(y1, x0, k0, x1, y0, k1)
    elif cfg["order"].val == 373:
        s[C].reorder(y1, x0, k0, x1, k1, y0)
    elif cfg["order"].val == 374:
        s[C].reorder(y1, x0, k0, y0, x1, k1)
    elif cfg["order"].val == 375:
        s[C].reorder(y1, x0, k0, y0, k1, x1)
    elif cfg["order"].val == 376:
        s[C].reorder(y1, x0, k0, k1, x1, y0)
    elif cfg["order"].val == 377:
        s[C].reorder(y1, x0, k0, k1, y0, x1)
    elif cfg["order"].val == 378:
        s[C].reorder(y1, x0, k1, x1, y0, k0)
    elif cfg["order"].val == 379:
        s[C].reorder(y1, x0, k1, x1, k0, y0)
    elif cfg["order"].val == 380:
        s[C].reorder(y1, x0, k1, y0, x1, k0)
    elif cfg["order"].val == 381:
        s[C].reorder(y1, x0, k1, y0, k0, x1)
    elif cfg["order"].val == 382:
        s[C].reorder(y1, x0, k1, k0, x1, y0)
    elif cfg["order"].val == 383:
        s[C].reorder(y1, x0, k1, k0, y0, x1)
    elif cfg["order"].val == 384:
        s[C].reorder(y1, x1, x0, y0, k0, k1)
    elif cfg["order"].val == 385:
        s[C].reorder(y1, x1, x0, y0, k1, k0)
    elif cfg["order"].val == 386:
        s[C].reorder(y1, x1, x0, k0, y0, k1)
    elif cfg["order"].val == 387:
        s[C].reorder(y1, x1, x0, k0, k1, y0)
    elif cfg["order"].val == 388:
        s[C].reorder(y1, x1, x0, k1, y0, k0)
    elif cfg["order"].val == 389:
        s[C].reorder(y1, x1, x0, k1, k0, y0)
    elif cfg["order"].val == 390:
        s[C].reorder(y1, x1, y0, x0, k0, k1)
    elif cfg["order"].val == 391:
        s[C].reorder(y1, x1, y0, x0, k1, k0)
    elif cfg["order"].val == 392:
        s[C].reorder(y1, x1, y0, k0, x0, k1)
    elif cfg["order"].val == 393:
        s[C].reorder(y1, x1, y0, k0, k1, x0)
    elif cfg["order"].val == 394:
        s[C].reorder(y1, x1, y0, k1, x0, k0)
    elif cfg["order"].val == 395:
        s[C].reorder(y1, x1, y0, k1, k0, x0)
    elif cfg["order"].val == 396:
        s[C].reorder(y1, x1, k0, x0, y0, k1)
    elif cfg["order"].val == 397:
        s[C].reorder(y1, x1, k0, x0, k1, y0)
    elif cfg["order"].val == 398:
        s[C].reorder(y1, x1, k0, y0, x0, k1)
    elif cfg["order"].val == 399:
        s[C].reorder(y1, x1, k0, y0, k1, x0)
    elif cfg["order"].val == 400:
        s[C].reorder(y1, x1, k0, k1, x0, y0)
    elif cfg["order"].val == 401:
        s[C].reorder(y1, x1, k0, k1, y0, x0)
    elif cfg["order"].val == 402:
        s[C].reorder(y1, x1, k1, x0, y0, k0)
    elif cfg["order"].val == 403:
        s[C].reorder(y1, x1, k1, x0, k0, y0)
    elif cfg["order"].val == 404:
        s[C].reorder(y1, x1, k1, y0, x0, k0)
    elif cfg["order"].val == 405:
        s[C].reorder(y1, x1, k1, y0, k0, x0)
    elif cfg["order"].val == 406:
        s[C].reorder(y1, x1, k1, k0, x0, y0)
    elif cfg["order"].val == 407:
        s[C].reorder(y1, x1, k1, k0, y0, x0)
    elif cfg["order"].val == 408:
        s[C].reorder(y1, y0, x0, x1, k0, k1)
    elif cfg["order"].val == 409:
        s[C].reorder(y1, y0, x0, x1, k1, k0)
    elif cfg["order"].val == 410:
        s[C].reorder(y1, y0, x0, k0, x1, k1)
    elif cfg["order"].val == 411:
        s[C].reorder(y1, y0, x0, k0, k1, x1)
    elif cfg["order"].val == 412:
        s[C].reorder(y1, y0, x0, k1, x1, k0)
    elif cfg["order"].val == 413:
        s[C].reorder(y1, y0, x0, k1, k0, x1)
    elif cfg["order"].val == 414:
        s[C].reorder(y1, y0, x1, x0, k0, k1)
    elif cfg["order"].val == 415:
        s[C].reorder(y1, y0, x1, x0, k1, k0)
    elif cfg["order"].val == 416:
        s[C].reorder(y1, y0, x1, k0, x0, k1)
    elif cfg["order"].val == 417:
        s[C].reorder(y1, y0, x1, k0, k1, x0)
    elif cfg["order"].val == 418:
        s[C].reorder(y1, y0, x1, k1, x0, k0)
    elif cfg["order"].val == 419:
        s[C].reorder(y1, y0, x1, k1, k0, x0)
    elif cfg["order"].val == 420:
        s[C].reorder(y1, y0, k0, x0, x1, k1)
    elif cfg["order"].val == 421:
        s[C].reorder(y1, y0, k0, x0, k1, x1)
    elif cfg["order"].val == 422:
        s[C].reorder(y1, y0, k0, x1, x0, k1)
    elif cfg["order"].val == 423:
        s[C].reorder(y1, y0, k0, x1, k1, x0)
    elif cfg["order"].val == 424:
        s[C].reorder(y1, y0, k0, k1, x0, x1)
    elif cfg["order"].val == 425:
        s[C].reorder(y1, y0, k0, k1, x1, x0)
    elif cfg["order"].val == 426:
        s[C].reorder(y1, y0, k1, x0, x1, k0)
    elif cfg["order"].val == 427:
        s[C].reorder(y1, y0, k1, x0, k0, x1)
    elif cfg["order"].val == 428:
        s[C].reorder(y1, y0, k1, x1, x0, k0)
    elif cfg["order"].val == 429:
        s[C].reorder(y1, y0, k1, x1, k0, x0)
    elif cfg["order"].val == 430:
        s[C].reorder(y1, y0, k1, k0, x0, x1)
    elif cfg["order"].val == 431:
        s[C].reorder(y1, y0, k1, k0, x1, x0)
    elif cfg["order"].val == 432:
        s[C].reorder(y1, k0, x0, x1, y0, k1)
    elif cfg["order"].val == 433:
        s[C].reorder(y1, k0, x0, x1, k1, y0)
    elif cfg["order"].val == 434:
        s[C].reorder(y1, k0, x0, y0, x1, k1)
    elif cfg["order"].val == 435:
        s[C].reorder(y1, k0, x0, y0, k1, x1)
    elif cfg["order"].val == 436:
        s[C].reorder(y1, k0, x0, k1, x1, y0)
    elif cfg["order"].val == 437:
        s[C].reorder(y1, k0, x0, k1, y0, x1)
    elif cfg["order"].val == 438:
        s[C].reorder(y1, k0, x1, x0, y0, k1)
    elif cfg["order"].val == 439:
        s[C].reorder(y1, k0, x1, x0, k1, y0)
    elif cfg["order"].val == 440:
        s[C].reorder(y1, k0, x1, y0, x0, k1)
    elif cfg["order"].val == 441:
        s[C].reorder(y1, k0, x1, y0, k1, x0)
    elif cfg["order"].val == 442:
        s[C].reorder(y1, k0, x1, k1, x0, y0)
    elif cfg["order"].val == 443:
        s[C].reorder(y1, k0, x1, k1, y0, x0)
    elif cfg["order"].val == 444:
        s[C].reorder(y1, k0, y0, x0, x1, k1)
    elif cfg["order"].val == 445:
        s[C].reorder(y1, k0, y0, x0, k1, x1)
    elif cfg["order"].val == 446:
        s[C].reorder(y1, k0, y0, x1, x0, k1)
    elif cfg["order"].val == 447:
        s[C].reorder(y1, k0, y0, x1, k1, x0)
    elif cfg["order"].val == 448:
        s[C].reorder(y1, k0, y0, k1, x0, x1)
    elif cfg["order"].val == 449:
        s[C].reorder(y1, k0, y0, k1, x1, x0)
    elif cfg["order"].val == 450:
        s[C].reorder(y1, k0, k1, x0, x1, y0)
    elif cfg["order"].val == 451:
        s[C].reorder(y1, k0, k1, x0, y0, x1)
    elif cfg["order"].val == 452:
        s[C].reorder(y1, k0, k1, x1, x0, y0)
    elif cfg["order"].val == 453:
        s[C].reorder(y1, k0, k1, x1, y0, x0)
    elif cfg["order"].val == 454:
        s[C].reorder(y1, k0, k1, y0, x0, x1)
    elif cfg["order"].val == 455:
        s[C].reorder(y1, k0, k1, y0, x1, x0)
    elif cfg["order"].val == 456:
        s[C].reorder(y1, k1, x0, x1, y0, k0)
    elif cfg["order"].val == 457:
        s[C].reorder(y1, k1, x0, x1, k0, y0)
    elif cfg["order"].val == 458:
        s[C].reorder(y1, k1, x0, y0, x1, k0)
    elif cfg["order"].val == 459:
        s[C].reorder(y1, k1, x0, y0, k0, x1)
    elif cfg["order"].val == 460:
        s[C].reorder(y1, k1, x0, k0, x1, y0)
    elif cfg["order"].val == 461:
        s[C].reorder(y1, k1, x0, k0, y0, x1)
    elif cfg["order"].val == 462:
        s[C].reorder(y1, k1, x1, x0, y0, k0)
    elif cfg["order"].val == 463:
        s[C].reorder(y1, k1, x1, x0, k0, y0)
    elif cfg["order"].val == 464:
        s[C].reorder(y1, k1, x1, y0, x0, k0)
    elif cfg["order"].val == 465:
        s[C].reorder(y1, k1, x1, y0, k0, x0)
    elif cfg["order"].val == 466:
        s[C].reorder(y1, k1, x1, k0, x0, y0)
    elif cfg["order"].val == 467:
        s[C].reorder(y1, k1, x1, k0, y0, x0)
    elif cfg["order"].val == 468:
        s[C].reorder(y1, k1, y0, x0, x1, k0)
    elif cfg["order"].val == 469:
        s[C].reorder(y1, k1, y0, x0, k0, x1)
    elif cfg["order"].val == 470:
        s[C].reorder(y1, k1, y0, x1, x0, k0)
    elif cfg["order"].val == 471:
        s[C].reorder(y1, k1, y0, x1, k0, x0)
    elif cfg["order"].val == 472:
        s[C].reorder(y1, k1, y0, k0, x0, x1)
    elif cfg["order"].val == 473:
        s[C].reorder(y1, k1, y0, k0, x1, x0)
    elif cfg["order"].val == 474:
        s[C].reorder(y1, k1, k0, x0, x1, y0)
    elif cfg["order"].val == 475:
        s[C].reorder(y1, k1, k0, x0, y0, x1)
    elif cfg["order"].val == 476:
        s[C].reorder(y1, k1, k0, x1, x0, y0)
    elif cfg["order"].val == 477:
        s[C].reorder(y1, k1, k0, x1, y0, x0)
    elif cfg["order"].val == 478:
        s[C].reorder(y1, k1, k0, y0, x0, x1)
    elif cfg["order"].val == 479:
        s[C].reorder(y1, k1, k0, y0, x1, x0)
    elif cfg["order"].val == 480:
        s[C].reorder(k0, x0, x1, y0, y1, k1)
    elif cfg["order"].val == 481:
        s[C].reorder(k0, x0, x1, y0, k1, y1)
    elif cfg["order"].val == 482:
        s[C].reorder(k0, x0, x1, y1, y0, k1)
    elif cfg["order"].val == 483:
        s[C].reorder(k0, x0, x1, y1, k1, y0)
    elif cfg["order"].val == 484:
        s[C].reorder(k0, x0, x1, k1, y0, y1)
    elif cfg["order"].val == 485:
        s[C].reorder(k0, x0, x1, k1, y1, y0)
    elif cfg["order"].val == 486:
        s[C].reorder(k0, x0, y0, x1, y1, k1)
    elif cfg["order"].val == 487:
        s[C].reorder(k0, x0, y0, x1, k1, y1)
    elif cfg["order"].val == 488:
        s[C].reorder(k0, x0, y0, y1, x1, k1)
    elif cfg["order"].val == 489:
        s[C].reorder(k0, x0, y0, y1, k1, x1)
    elif cfg["order"].val == 490:
        s[C].reorder(k0, x0, y0, k1, x1, y1)
    elif cfg["order"].val == 491:
        s[C].reorder(k0, x0, y0, k1, y1, x1)
    elif cfg["order"].val == 492:
        s[C].reorder(k0, x0, y1, x1, y0, k1)
    elif cfg["order"].val == 493:
        s[C].reorder(k0, x0, y1, x1, k1, y0)
    elif cfg["order"].val == 494:
        s[C].reorder(k0, x0, y1, y0, x1, k1)
    elif cfg["order"].val == 495:
        s[C].reorder(k0, x0, y1, y0, k1, x1)
    elif cfg["order"].val == 496:
        s[C].reorder(k0, x0, y1, k1, x1, y0)
    elif cfg["order"].val == 497:
        s[C].reorder(k0, x0, y1, k1, y0, x1)
    elif cfg["order"].val == 498:
        s[C].reorder(k0, x0, k1, x1, y0, y1)
    elif cfg["order"].val == 499:
        s[C].reorder(k0, x0, k1, x1, y1, y0)
    elif cfg["order"].val == 500:
        s[C].reorder(k0, x0, k1, y0, x1, y1)
    elif cfg["order"].val == 501:
        s[C].reorder(k0, x0, k1, y0, y1, x1)
    elif cfg["order"].val == 502:
        s[C].reorder(k0, x0, k1, y1, x1, y0)
    elif cfg["order"].val == 503:
        s[C].reorder(k0, x0, k1, y1, y0, x1)
    elif cfg["order"].val == 504:
        s[C].reorder(k0, x1, x0, y0, y1, k1)
    elif cfg["order"].val == 505:
        s[C].reorder(k0, x1, x0, y0, k1, y1)
    elif cfg["order"].val == 506:
        s[C].reorder(k0, x1, x0, y1, y0, k1)
    elif cfg["order"].val == 507:
        s[C].reorder(k0, x1, x0, y1, k1, y0)
    elif cfg["order"].val == 508:
        s[C].reorder(k0, x1, x0, k1, y0, y1)
    elif cfg["order"].val == 509:
        s[C].reorder(k0, x1, x0, k1, y1, y0)
    elif cfg["order"].val == 510:
        s[C].reorder(k0, x1, y0, x0, y1, k1)
    elif cfg["order"].val == 511:
        s[C].reorder(k0, x1, y0, x0, k1, y1)
    elif cfg["order"].val == 512:
        s[C].reorder(k0, x1, y0, y1, x0, k1)
    elif cfg["order"].val == 513:
        s[C].reorder(k0, x1, y0, y1, k1, x0)
    elif cfg["order"].val == 514:
        s[C].reorder(k0, x1, y0, k1, x0, y1)
    elif cfg["order"].val == 515:
        s[C].reorder(k0, x1, y0, k1, y1, x0)
    elif cfg["order"].val == 516:
        s[C].reorder(k0, x1, y1, x0, y0, k1)
    elif cfg["order"].val == 517:
        s[C].reorder(k0, x1, y1, x0, k1, y0)
    elif cfg["order"].val == 518:
        s[C].reorder(k0, x1, y1, y0, x0, k1)
    elif cfg["order"].val == 519:
        s[C].reorder(k0, x1, y1, y0, k1, x0)
    elif cfg["order"].val == 520:
        s[C].reorder(k0, x1, y1, k1, x0, y0)
    elif cfg["order"].val == 521:
        s[C].reorder(k0, x1, y1, k1, y0, x0)
    elif cfg["order"].val == 522:
        s[C].reorder(k0, x1, k1, x0, y0, y1)
    elif cfg["order"].val == 523:
        s[C].reorder(k0, x1, k1, x0, y1, y0)
    elif cfg["order"].val == 524:
        s[C].reorder(k0, x1, k1, y0, x0, y1)
    elif cfg["order"].val == 525:
        s[C].reorder(k0, x1, k1, y0, y1, x0)
    elif cfg["order"].val == 526:
        s[C].reorder(k0, x1, k1, y1, x0, y0)
    elif cfg["order"].val == 527:
        s[C].reorder(k0, x1, k1, y1, y0, x0)
    elif cfg["order"].val == 528:
        s[C].reorder(k0, y0, x0, x1, y1, k1)
    elif cfg["order"].val == 529:
        s[C].reorder(k0, y0, x0, x1, k1, y1)
    elif cfg["order"].val == 530:
        s[C].reorder(k0, y0, x0, y1, x1, k1)
    elif cfg["order"].val == 531:
        s[C].reorder(k0, y0, x0, y1, k1, x1)
    elif cfg["order"].val == 532:
        s[C].reorder(k0, y0, x0, k1, x1, y1)
    elif cfg["order"].val == 533:
        s[C].reorder(k0, y0, x0, k1, y1, x1)
    elif cfg["order"].val == 534:
        s[C].reorder(k0, y0, x1, x0, y1, k1)
    elif cfg["order"].val == 535:
        s[C].reorder(k0, y0, x1, x0, k1, y1)
    elif cfg["order"].val == 536:
        s[C].reorder(k0, y0, x1, y1, x0, k1)
    elif cfg["order"].val == 537:
        s[C].reorder(k0, y0, x1, y1, k1, x0)
    elif cfg["order"].val == 538:
        s[C].reorder(k0, y0, x1, k1, x0, y1)
    elif cfg["order"].val == 539:
        s[C].reorder(k0, y0, x1, k1, y1, x0)
    elif cfg["order"].val == 540:
        s[C].reorder(k0, y0, y1, x0, x1, k1)
    elif cfg["order"].val == 541:
        s[C].reorder(k0, y0, y1, x0, k1, x1)
    elif cfg["order"].val == 542:
        s[C].reorder(k0, y0, y1, x1, x0, k1)
    elif cfg["order"].val == 543:
        s[C].reorder(k0, y0, y1, x1, k1, x0)
    elif cfg["order"].val == 544:
        s[C].reorder(k0, y0, y1, k1, x0, x1)
    elif cfg["order"].val == 545:
        s[C].reorder(k0, y0, y1, k1, x1, x0)
    elif cfg["order"].val == 546:
        s[C].reorder(k0, y0, k1, x0, x1, y1)
    elif cfg["order"].val == 547:
        s[C].reorder(k0, y0, k1, x0, y1, x1)
    elif cfg["order"].val == 548:
        s[C].reorder(k0, y0, k1, x1, x0, y1)
    elif cfg["order"].val == 549:
        s[C].reorder(k0, y0, k1, x1, y1, x0)
    elif cfg["order"].val == 550:
        s[C].reorder(k0, y0, k1, y1, x0, x1)
    elif cfg["order"].val == 551:
        s[C].reorder(k0, y0, k1, y1, x1, x0)
    elif cfg["order"].val == 552:
        s[C].reorder(k0, y1, x0, x1, y0, k1)
    elif cfg["order"].val == 553:
        s[C].reorder(k0, y1, x0, x1, k1, y0)
    elif cfg["order"].val == 554:
        s[C].reorder(k0, y1, x0, y0, x1, k1)
    elif cfg["order"].val == 555:
        s[C].reorder(k0, y1, x0, y0, k1, x1)
    elif cfg["order"].val == 556:
        s[C].reorder(k0, y1, x0, k1, x1, y0)
    elif cfg["order"].val == 557:
        s[C].reorder(k0, y1, x0, k1, y0, x1)
    elif cfg["order"].val == 558:
        s[C].reorder(k0, y1, x1, x0, y0, k1)
    elif cfg["order"].val == 559:
        s[C].reorder(k0, y1, x1, x0, k1, y0)
    elif cfg["order"].val == 560:
        s[C].reorder(k0, y1, x1, y0, x0, k1)
    elif cfg["order"].val == 561:
        s[C].reorder(k0, y1, x1, y0, k1, x0)
    elif cfg["order"].val == 562:
        s[C].reorder(k0, y1, x1, k1, x0, y0)
    elif cfg["order"].val == 563:
        s[C].reorder(k0, y1, x1, k1, y0, x0)
    elif cfg["order"].val == 564:
        s[C].reorder(k0, y1, y0, x0, x1, k1)
    elif cfg["order"].val == 565:
        s[C].reorder(k0, y1, y0, x0, k1, x1)
    elif cfg["order"].val == 566:
        s[C].reorder(k0, y1, y0, x1, x0, k1)
    elif cfg["order"].val == 567:
        s[C].reorder(k0, y1, y0, x1, k1, x0)
    elif cfg["order"].val == 568:
        s[C].reorder(k0, y1, y0, k1, x0, x1)
    elif cfg["order"].val == 569:
        s[C].reorder(k0, y1, y0, k1, x1, x0)
    elif cfg["order"].val == 570:
        s[C].reorder(k0, y1, k1, x0, x1, y0)
    elif cfg["order"].val == 571:
        s[C].reorder(k0, y1, k1, x0, y0, x1)
    elif cfg["order"].val == 572:
        s[C].reorder(k0, y1, k1, x1, x0, y0)
    elif cfg["order"].val == 573:
        s[C].reorder(k0, y1, k1, x1, y0, x0)
    elif cfg["order"].val == 574:
        s[C].reorder(k0, y1, k1, y0, x0, x1)
    elif cfg["order"].val == 575:
        s[C].reorder(k0, y1, k1, y0, x1, x0)
    elif cfg["order"].val == 576:
        s[C].reorder(k0, k1, x0, x1, y0, y1)
    elif cfg["order"].val == 577:
        s[C].reorder(k0, k1, x0, x1, y1, y0)
    elif cfg["order"].val == 578:
        s[C].reorder(k0, k1, x0, y0, x1, y1)
    elif cfg["order"].val == 579:
        s[C].reorder(k0, k1, x0, y0, y1, x1)
    elif cfg["order"].val == 580:
        s[C].reorder(k0, k1, x0, y1, x1, y0)
    elif cfg["order"].val == 581:
        s[C].reorder(k0, k1, x0, y1, y0, x1)
    elif cfg["order"].val == 582:
        s[C].reorder(k0, k1, x1, x0, y0, y1)
    elif cfg["order"].val == 583:
        s[C].reorder(k0, k1, x1, x0, y1, y0)
    elif cfg["order"].val == 584:
        s[C].reorder(k0, k1, x1, y0, x0, y1)
    elif cfg["order"].val == 585:
        s[C].reorder(k0, k1, x1, y0, y1, x0)
    elif cfg["order"].val == 586:
        s[C].reorder(k0, k1, x1, y1, x0, y0)
    elif cfg["order"].val == 587:
        s[C].reorder(k0, k1, x1, y1, y0, x0)
    elif cfg["order"].val == 588:
        s[C].reorder(k0, k1, y0, x0, x1, y1)
    elif cfg["order"].val == 589:
        s[C].reorder(k0, k1, y0, x0, y1, x1)
    elif cfg["order"].val == 590:
        s[C].reorder(k0, k1, y0, x1, x0, y1)
    elif cfg["order"].val == 591:
        s[C].reorder(k0, k1, y0, x1, y1, x0)
    elif cfg["order"].val == 592:
        s[C].reorder(k0, k1, y0, y1, x0, x1)
    elif cfg["order"].val == 593:
        s[C].reorder(k0, k1, y0, y1, x1, x0)
    elif cfg["order"].val == 594:
        s[C].reorder(k0, k1, y1, x0, x1, y0)
    elif cfg["order"].val == 595:
        s[C].reorder(k0, k1, y1, x0, y0, x1)
    elif cfg["order"].val == 596:
        s[C].reorder(k0, k1, y1, x1, x0, y0)
    elif cfg["order"].val == 597:
        s[C].reorder(k0, k1, y1, x1, y0, x0)
    elif cfg["order"].val == 598:
        s[C].reorder(k0, k1, y1, y0, x0, x1)
    elif cfg["order"].val == 599:
        s[C].reorder(k0, k1, y1, y0, x1, x0)
    elif cfg["order"].val == 600:
        s[C].reorder(k1, x0, x1, y0, y1, k0)
    elif cfg["order"].val == 601:
        s[C].reorder(k1, x0, x1, y0, k0, y1)
    elif cfg["order"].val == 602:
        s[C].reorder(k1, x0, x1, y1, y0, k0)
    elif cfg["order"].val == 603:
        s[C].reorder(k1, x0, x1, y1, k0, y0)
    elif cfg["order"].val == 604:
        s[C].reorder(k1, x0, x1, k0, y0, y1)
    elif cfg["order"].val == 605:
        s[C].reorder(k1, x0, x1, k0, y1, y0)
    elif cfg["order"].val == 606:
        s[C].reorder(k1, x0, y0, x1, y1, k0)
    elif cfg["order"].val == 607:
        s[C].reorder(k1, x0, y0, x1, k0, y1)
    elif cfg["order"].val == 608:
        s[C].reorder(k1, x0, y0, y1, x1, k0)
    elif cfg["order"].val == 609:
        s[C].reorder(k1, x0, y0, y1, k0, x1)
    elif cfg["order"].val == 610:
        s[C].reorder(k1, x0, y0, k0, x1, y1)
    elif cfg["order"].val == 611:
        s[C].reorder(k1, x0, y0, k0, y1, x1)
    elif cfg["order"].val == 612:
        s[C].reorder(k1, x0, y1, x1, y0, k0)
    elif cfg["order"].val == 613:
        s[C].reorder(k1, x0, y1, x1, k0, y0)
    elif cfg["order"].val == 614:
        s[C].reorder(k1, x0, y1, y0, x1, k0)
    elif cfg["order"].val == 615:
        s[C].reorder(k1, x0, y1, y0, k0, x1)
    elif cfg["order"].val == 616:
        s[C].reorder(k1, x0, y1, k0, x1, y0)
    elif cfg["order"].val == 617:
        s[C].reorder(k1, x0, y1, k0, y0, x1)
    elif cfg["order"].val == 618:
        s[C].reorder(k1, x0, k0, x1, y0, y1)
    elif cfg["order"].val == 619:
        s[C].reorder(k1, x0, k0, x1, y1, y0)
    elif cfg["order"].val == 620:
        s[C].reorder(k1, x0, k0, y0, x1, y1)
    elif cfg["order"].val == 621:
        s[C].reorder(k1, x0, k0, y0, y1, x1)
    elif cfg["order"].val == 622:
        s[C].reorder(k1, x0, k0, y1, x1, y0)
    elif cfg["order"].val == 623:
        s[C].reorder(k1, x0, k0, y1, y0, x1)
    elif cfg["order"].val == 624:
        s[C].reorder(k1, x1, x0, y0, y1, k0)
    elif cfg["order"].val == 625:
        s[C].reorder(k1, x1, x0, y0, k0, y1)
    elif cfg["order"].val == 626:
        s[C].reorder(k1, x1, x0, y1, y0, k0)
    elif cfg["order"].val == 627:
        s[C].reorder(k1, x1, x0, y1, k0, y0)
    elif cfg["order"].val == 628:
        s[C].reorder(k1, x1, x0, k0, y0, y1)
    elif cfg["order"].val == 629:
        s[C].reorder(k1, x1, x0, k0, y1, y0)
    elif cfg["order"].val == 630:
        s[C].reorder(k1, x1, y0, x0, y1, k0)
    elif cfg["order"].val == 631:
        s[C].reorder(k1, x1, y0, x0, k0, y1)
    elif cfg["order"].val == 632:
        s[C].reorder(k1, x1, y0, y1, x0, k0)
    elif cfg["order"].val == 633:
        s[C].reorder(k1, x1, y0, y1, k0, x0)
    elif cfg["order"].val == 634:
        s[C].reorder(k1, x1, y0, k0, x0, y1)
    elif cfg["order"].val == 635:
        s[C].reorder(k1, x1, y0, k0, y1, x0)
    elif cfg["order"].val == 636:
        s[C].reorder(k1, x1, y1, x0, y0, k0)
    elif cfg["order"].val == 637:
        s[C].reorder(k1, x1, y1, x0, k0, y0)
    elif cfg["order"].val == 638:
        s[C].reorder(k1, x1, y1, y0, x0, k0)
    elif cfg["order"].val == 639:
        s[C].reorder(k1, x1, y1, y0, k0, x0)
    elif cfg["order"].val == 640:
        s[C].reorder(k1, x1, y1, k0, x0, y0)
    elif cfg["order"].val == 641:
        s[C].reorder(k1, x1, y1, k0, y0, x0)
    elif cfg["order"].val == 642:
        s[C].reorder(k1, x1, k0, x0, y0, y1)
    elif cfg["order"].val == 643:
        s[C].reorder(k1, x1, k0, x0, y1, y0)
    elif cfg["order"].val == 644:
        s[C].reorder(k1, x1, k0, y0, x0, y1)
    elif cfg["order"].val == 645:
        s[C].reorder(k1, x1, k0, y0, y1, x0)
    elif cfg["order"].val == 646:
        s[C].reorder(k1, x1, k0, y1, x0, y0)
    elif cfg["order"].val == 647:
        s[C].reorder(k1, x1, k0, y1, y0, x0)
    elif cfg["order"].val == 648:
        s[C].reorder(k1, y0, x0, x1, y1, k0)
    elif cfg["order"].val == 649:
        s[C].reorder(k1, y0, x0, x1, k0, y1)
    elif cfg["order"].val == 650:
        s[C].reorder(k1, y0, x0, y1, x1, k0)
    elif cfg["order"].val == 651:
        s[C].reorder(k1, y0, x0, y1, k0, x1)
    elif cfg["order"].val == 652:
        s[C].reorder(k1, y0, x0, k0, x1, y1)
    elif cfg["order"].val == 653:
        s[C].reorder(k1, y0, x0, k0, y1, x1)
    elif cfg["order"].val == 654:
        s[C].reorder(k1, y0, x1, x0, y1, k0)
    elif cfg["order"].val == 655:
        s[C].reorder(k1, y0, x1, x0, k0, y1)
    elif cfg["order"].val == 656:
        s[C].reorder(k1, y0, x1, y1, x0, k0)
    elif cfg["order"].val == 657:
        s[C].reorder(k1, y0, x1, y1, k0, x0)
    elif cfg["order"].val == 658:
        s[C].reorder(k1, y0, x1, k0, x0, y1)
    elif cfg["order"].val == 659:
        s[C].reorder(k1, y0, x1, k0, y1, x0)
    elif cfg["order"].val == 660:
        s[C].reorder(k1, y0, y1, x0, x1, k0)
    elif cfg["order"].val == 661:
        s[C].reorder(k1, y0, y1, x0, k0, x1)
    elif cfg["order"].val == 662:
        s[C].reorder(k1, y0, y1, x1, x0, k0)
    elif cfg["order"].val == 663:
        s[C].reorder(k1, y0, y1, x1, k0, x0)
    elif cfg["order"].val == 664:
        s[C].reorder(k1, y0, y1, k0, x0, x1)
    elif cfg["order"].val == 665:
        s[C].reorder(k1, y0, y1, k0, x1, x0)
    elif cfg["order"].val == 666:
        s[C].reorder(k1, y0, k0, x0, x1, y1)
    elif cfg["order"].val == 667:
        s[C].reorder(k1, y0, k0, x0, y1, x1)
    elif cfg["order"].val == 668:
        s[C].reorder(k1, y0, k0, x1, x0, y1)
    elif cfg["order"].val == 669:
        s[C].reorder(k1, y0, k0, x1, y1, x0)
    elif cfg["order"].val == 670:
        s[C].reorder(k1, y0, k0, y1, x0, x1)
    elif cfg["order"].val == 671:
        s[C].reorder(k1, y0, k0, y1, x1, x0)
    elif cfg["order"].val == 672:
        s[C].reorder(k1, y1, x0, x1, y0, k0)
    elif cfg["order"].val == 673:
        s[C].reorder(k1, y1, x0, x1, k0, y0)
    elif cfg["order"].val == 674:
        s[C].reorder(k1, y1, x0, y0, x1, k0)
    elif cfg["order"].val == 675:
        s[C].reorder(k1, y1, x0, y0, k0, x1)
    elif cfg["order"].val == 676:
        s[C].reorder(k1, y1, x0, k0, x1, y0)
    elif cfg["order"].val == 677:
        s[C].reorder(k1, y1, x0, k0, y0, x1)
    elif cfg["order"].val == 678:
        s[C].reorder(k1, y1, x1, x0, y0, k0)
    elif cfg["order"].val == 679:
        s[C].reorder(k1, y1, x1, x0, k0, y0)
    elif cfg["order"].val == 680:
        s[C].reorder(k1, y1, x1, y0, x0, k0)
    elif cfg["order"].val == 681:
        s[C].reorder(k1, y1, x1, y0, k0, x0)
    elif cfg["order"].val == 682:
        s[C].reorder(k1, y1, x1, k0, x0, y0)
    elif cfg["order"].val == 683:
        s[C].reorder(k1, y1, x1, k0, y0, x0)
    elif cfg["order"].val == 684:
        s[C].reorder(k1, y1, y0, x0, x1, k0)
    elif cfg["order"].val == 685:
        s[C].reorder(k1, y1, y0, x0, k0, x1)
    elif cfg["order"].val == 686:
        s[C].reorder(k1, y1, y0, x1, x0, k0)
    elif cfg["order"].val == 687:
        s[C].reorder(k1, y1, y0, x1, k0, x0)
    elif cfg["order"].val == 688:
        s[C].reorder(k1, y1, y0, k0, x0, x1)
    elif cfg["order"].val == 689:
        s[C].reorder(k1, y1, y0, k0, x1, x0)
    elif cfg["order"].val == 690:
        s[C].reorder(k1, y1, k0, x0, x1, y0)
    elif cfg["order"].val == 691:
        s[C].reorder(k1, y1, k0, x0, y0, x1)
    elif cfg["order"].val == 692:
        s[C].reorder(k1, y1, k0, x1, x0, y0)
    elif cfg["order"].val == 693:
        s[C].reorder(k1, y1, k0, x1, y0, x0)
    elif cfg["order"].val == 694:
        s[C].reorder(k1, y1, k0, y0, x0, x1)
    elif cfg["order"].val == 695:
        s[C].reorder(k1, y1, k0, y0, x1, x0)
    elif cfg["order"].val == 696:
        s[C].reorder(k1, k0, x0, x1, y0, y1)
    elif cfg["order"].val == 697:
        s[C].reorder(k1, k0, x0, x1, y1, y0)
    elif cfg["order"].val == 698:
        s[C].reorder(k1, k0, x0, y0, x1, y1)
    elif cfg["order"].val == 699:
        s[C].reorder(k1, k0, x0, y0, y1, x1)
    elif cfg["order"].val == 700:
        s[C].reorder(k1, k0, x0, y1, x1, y0)
    elif cfg["order"].val == 701:
        s[C].reorder(k1, k0, x0, y1, y0, x1)
    elif cfg["order"].val == 702:
        s[C].reorder(k1, k0, x1, x0, y0, y1)
    elif cfg["order"].val == 703:
        s[C].reorder(k1, k0, x1, x0, y1, y0)
    elif cfg["order"].val == 704:
        s[C].reorder(k1, k0, x1, y0, x0, y1)
    elif cfg["order"].val == 705:
        s[C].reorder(k1, k0, x1, y0, y1, x0)
    elif cfg["order"].val == 706:
        s[C].reorder(k1, k0, x1, y1, x0, y0)
    elif cfg["order"].val == 707:
        s[C].reorder(k1, k0, x1, y1, y0, x0)
    elif cfg["order"].val == 708:
        s[C].reorder(k1, k0, y0, x0, x1, y1)
    elif cfg["order"].val == 709:
        s[C].reorder(k1, k0, y0, x0, y1, x1)
    elif cfg["order"].val == 710:
        s[C].reorder(k1, k0, y0, x1, x0, y1)
    elif cfg["order"].val == 711:
        s[C].reorder(k1, k0, y0, x1, y1, x0)
    elif cfg["order"].val == 712:
        s[C].reorder(k1, k0, y0, y1, x0, x1)
    elif cfg["order"].val == 713:
        s[C].reorder(k1, k0, y0, y1, x1, x0)
    elif cfg["order"].val == 714:
        s[C].reorder(k1, k0, y1, x0, x1, y0)
    elif cfg["order"].val == 715:
        s[C].reorder(k1, k0, y1, x0, y0, x1)
    elif cfg["order"].val == 716:
        s[C].reorder(k1, k0, y1, x1, x0, y0)
    elif cfg["order"].val == 717:
        s[C].reorder(k1, k0, y1, x1, y0, x0)
    elif cfg["order"].val == 718:
        s[C].reorder(k1, k0, y1, y0, x0, x1)
    elif cfg["order"].val == 719:
        s[C].reorder(k1, k0, y1, y0, x1, x0)

    cfg.define_knob("unroll", ["None","x0", "x1", "y0", "y1", "k0", "k1", "x0x1", "x0y0", "x0y1", "x0k0", "x0k1", "x1y0", "x1y1", "x1k0", "x1k1", "y0y1", "y0k0", "y0k1", "y1k0", "y1k1", "k0k1", "x0x1y0", "x0x1y1", "x0x1k0", "x0x1k1", "x0y0y1", "x0y0k0", "x0y0k1", "x0y1k0", "x0y1k1", "x0k0k1", "x1y0y1", "x1y0k0", "x1y0k1", "x1y1k0", "x1y1k1", "x1k0k1", "y0y1k0", "y0y1k1", "y0k0k1", "y1k0k1", "x0x1y0y1", "x0x1y0k0", "x0x1y0k1", "x0x1y1k0", "x0x1y1k1", "x0x1k0k1", "x0y0y1k0", "x0y0y1k1", "x0y0k0k1", "x0y1k0k1", "x1y0y1k0", "x1y0y1k1", "x1y0k0k1", "x1y1k0k1", "y0y1k0k1", "x0x1y0y1k0", "x0x1y0y1k1", "x0x1y0k0k1", "x0x1y1k0k1", "x0y0y1k0k1", "x1y0y1k0k1", "x0x1y0y1k0k1"])

    if "x0" in cfg["unroll"].val:
        s[C].unroll(x0)
    if "x1" in cfg["unroll"].val:
        s[C].unroll(x1)
    if "y0" in cfg["unroll"].val:
        s[C].unroll(y0)
    if "y1" in cfg["unroll"].val:
        s[C].unroll(y1)
    if "k0" in cfg["unroll"].val:
        s[C].unroll(k0)
    if "k1" in cfg["unroll"].val:
        s[C].unroll(k1)

    cfg.define_knob("vec", ["None","x0", "x1", "y0", "y1", "k0", "k1", "x0x1", "x0y0", "x0y1", "x0k0", "x0k1", "x1y0", "x1y1", "x1k0", "x1k1", "y0y1", "y0k0", "y0k1", "y1k0", "y1k1", "k0k1", "x0x1y0", "x0x1y1", "x0x1k0", "x0x1k1", "x0y0y1", "x0y0k0", "x0y0k1", "x0y1k0", "x0y1k1", "x0k0k1", "x1y0y1", "x1y0k0", "x1y0k1", "x1y1k0", "x1y1k1", "x1k0k1", "y0y1k0", "y0y1k1", "y0k0k1", "y1k0k1", "x0x1y0y1", "x0x1y0k0", "x0x1y0k1", "x0x1y1k0", "x0x1y1k1", "x0x1k0k1", "x0y0y1k0", "x0y0y1k1", "x0y0k0k1", "x0y1k0k1", "x1y0y1k0", "x1y0y1k1", "x1y0k0k1", "x1y1k0k1", "y0y1k0k1", "x0x1y0y1k0", "x0x1y0y1k1", "x0x1y0k0k1", "x0x1y1k0k1", "x0y0y1k0k1", "x1y0y1k0k1", "x0x1y0y1k0k1"])
    
    if "x0" in cfg["vec"].val:
        s[C].vectorize(x0)
    if "x1" in cfg["vec"].val:
        s[C].vectorize(x1)
    if "y0" in cfg["vec"].val:
        s[C].vectorize(y0)
    if "y1" in cfg["vec"].val:
        s[C].vectorize(y1)
    if "k0" in cfg["vec"].val:
        s[C].vectorize(k0)
    if "k1" in cfg["vec"].val:
        s[C].vectorize(k1)

    return s, [A, B, C]

if __name__ == "__main__":

    N, L, M = 1000, 800, 700
    search_space_tile = [1] + [i for i in range(8,257,8)]

    target = tvm.target.Target("cuda")
    dev = tvm.cuda()

    np.random.seed(0)
    a_np = np.random.uniform(size=(N, L)).astype(np.float32)
    b_np = np.random.uniform(size=(L, M)).astype(np.float32)
    c_np = a_np.dot(b_np)

    #tool = ["DropletTuner", "GridSearchTuner", "RandomTuner", "GATuner", "XGBTuner"]
    tool = ["DropletTuner"]

    for t in tool:

        save_log = "results/%s_mm.log" % (t)
        if os.path.isfile(save_log):
            os.remove(save_log)

        task = autotvm.task.create("template_matmul", args=(N, L, M, search_space_tile, "float32"), target=target)

        print(task.config_space)

        #logging.getLogger("autotvm").setLevel(logging.disable)
        #logging.getLogger("autotvm").addHandler(logging.StreamHandler(sys.stdout))

        measure_option = autotvm.measure_option(builder="local", runner=autotvm.LocalRunner(number=2, repeat=3))

        start = time.time()
            
        if t == "DropletTuner":
            n_trial = 2000
            tuner = autotvm.tuner.DropletTuner(task)
        elif t == "GridSearchTuner":
            n_trial = len(task.config_space)            # 100% of search space
            tuner = autotvm.tuner.GridSearchTuner(task)
        elif t == "RandomTuner":
            n_trial = int(len(task.config_space) * 0.3) # %30% of search space
            tuner = autotvm.tuner.RandomTuner(task)
        elif t == "GATuner":
            n_trial = int(len(task.config_space) * 0.3) # %30% of search space
            tuner = autotvm.tuner.GATuner(task)
        elif t == "XGBTuner":
            n_trial = int(len(task.config_space) * 0.3) # %30% of search space
            tuner = autotvm.tuner.XGBTuner(task, loss_type="rank")

        tuner.tune(
            n_trial=n_trial,
            measure_option=measure_option,
            callbacks=[autotvm.callback.log_to_file(save_log)],
        )

        end = time.time()

        # inspect the best config
        dispatch_context = autotvm.apply_history_best(save_log)
        best_config = dispatch_context.query(task.target, task.workload)
        print("%s, Best config:" %(t), best_config, end="")

        # apply history best from log file
        with autotvm.apply_history_best(save_log):
            with tvm.target.Target(target):
                s, arg_bufs = matmul(N, L, M, search_space_tile, "float32")
                func = tvm.build(s, arg_bufs)

        # check correctness
        # check correctness
        a_tvm = tvm.nd.array(a_np, device=dev)
        b_tvm = tvm.nd.array(b_np, device=dev)
        c_tvm = tvm.nd.empty(c_np.shape, device=dev)
        func(a_tvm, b_tvm, c_tvm)

        tvm.testing.assert_allclose(c_np, c_tvm.numpy(), rtol=1e-4)

        # Evaluate running time. Here we choose a large repeat number (400) to reduce the noise
        # and the overhead of kernel launch. y0u can also use nvprof to validate the result.
        evaluator = func.time_evaluator(func.entry_name, dev, number=10, repeat=3)
        eval = evaluator(a_tvm, b_tvm, c_tvm)
        print(", %f, %f, %f" % (eval.mean, eval.std, end-start))

        break
