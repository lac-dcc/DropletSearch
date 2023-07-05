import tvm
from tvm import relay, autotvm
from tvm.relay import testing
from tvm.contrib import graph_runtime
import numpy as np

mod, params = testing.lstm.get_workload(iterations=4, num_hidden=10, batch_size=1, dtype="float32")

#print(mod["main"])

tasks = autotvm.task.extract_from_program(mod["main"], target="llvm", params=params, ops=(
    relay.op.get("nn.dense"),
    relay.op.get("nn.bias_add"),
    )
)

for i, t in enumerate(tasks):
    print(i, t)

'''
vs = relay.analysis.all_vars(mod["main"])
tp = dict()
for idx, v in enumerate(vs):
    if "input" not in str(v.name_hint):
        shape = [int(_) for _ in v.type_annotation.shape]
        p = np.ones(shape).astype(np.int32) * idx
        tp[str(v.name_hint)] = tvm.nd.array(p)
lib = relay.build(mod['main'], target="llvm", params=tp)
'''
