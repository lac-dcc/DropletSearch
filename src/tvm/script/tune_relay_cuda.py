# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""
.. _tune_relay_x86:

Auto-tuning a Convolutional Network for Cuda
===============================================
**Author**: `Yao Wang <https://github.com/kevinthesun>`_, `Eddie Yan <https://github.com/eqy>`_

This is a tutorial about how to tune convolution neural network
for Cuda.
"""

import os
import numpy as np
import time

import tvm
from tvm import relay, autotvm
from tvm.relay import testing
from tvm.autotvm.tuner import XGBTuner, GATuner, RandomTuner, GridSearchTuner, DropletTuner
from tvm.autotvm.graph_tuner import DPTuner, PBQPTuner
import tvm.contrib.graph_executor as runtime

#################################################################
# Define network
# --------------
target = tvm.target.cuda()

batch_size = 1
dtype = "float32"
model_name = "resnet-18"

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

graph_opt_sch_file = "%s_graph_opt.log" % model_name

# Set the input name of the graph
# For ONNX models, it is typically "0".
input_name = "data"

# Set number of threads used for tuning based on the number of
# physical CPU cores on your machine.
num_threads = os.cpu_count()
os.environ["TVM_NUM_THREADS"] = str(num_threads)

def get_network(name, batch_size):
    """Get the symbol definition and random weight of a network"""
    input_shape = (batch_size, 3, 224, 224)
    output_shape = (batch_size, 1000)

    if "resnet" in name:
        n_layer = int(name.split("-")[1])
        mod, params = relay.testing.resnet.get_workload(
            num_layers=n_layer, batch_size=batch_size, dtype=dtype
        )
    elif "vgg" in name:
        n_layer = int(name.split("-")[1])
        mod, params = relay.testing.vgg.get_workload(
            num_layers=n_layer, batch_size=batch_size, dtype=dtype
        )
    elif name == "mobilenet":
        mod, params = relay.testing.mobilenet.get_workload(batch_size=batch_size, dtype=dtype)
    elif name == "squeezenet_v1.1":
        mod, params = relay.testing.squeezenet.get_workload(
            batch_size=batch_size, version="1.1", dtype=dtype
        )
    elif name == "inception_v3":
        input_shape = (batch_size, 3, 299, 299)
        mod, params = relay.testing.inception_v3.get_workload(batch_size=batch_size, dtype=dtype)
    elif name == "mxnet":
        # an example for mxnet model
        from mxnet.gluon.model_zoo.vision import get_model

        block = get_model("resnet18_v1", pretrained=True)
        mod, params = relay.frontend.from_mxnet(block, shape={input_name: input_shape}, dtype=dtype)
        net = mod["main"]
        net = relay.Function(
            net.params, relay.nn.softmax(net.body), None, net.type_params, net.attrs
        )
        mod = tvm.IRModule.from_expr(net)
    else:
        raise ValueError("Unsupported network: " + name)

    return mod, params, input_shape, output_shape

#################################################################
# Configure tensor tuning settings and create tasks
# -------------------------------------------------
# To get better kernel execution performance on x86 CPU,
# we need to change data layout of convolution kernel from
# "NCHW" to "NCHWc". To deal with this situation, we define
# conv2d_NCHWc operator in topi. We will tune this operator
# instead of plain conv2d.
#
# We will use local mode for tuning configuration. RPC tracker
# mode can be setup similarly to the approach in
# :ref:`tune_relay_arm` tutorial.
#
# To perform a precise measurement, we should repeat the measurement several
# times and use the average of results. In addition, we need to flush the cache
# for the weight tensors between repeated measurements. This can make the measured
# latency of one operator closer to its actual latency during end-to-end inference.

# You can skip the implementation of this function for this tutorial.
def tune_kernels(
    tasks, measure_option, tuner="gridsearch", early_stopping=None, log_filename="tuning.log"
):
    for i, task in enumerate(tasks):
        prefix = "[Task %2d/%2d] " % (i + 1, len(tasks))

        log_filename_tmp = log_filename + "_layer_" + str(i) + ".log"

        if os.path.exists(log_filename_tmp):
            os.remove(log_filename_tmp)

        print("-" * 40)
        print("Layer: %d" %i)
        print("-" * 40)
        print(task.args)
        #print(task.args)
        for k, v in task.config_space.space_map.items():
            print(k, end="")
            for l in v:
                print(",", l, end="")
            print()
        
        # create tuner
        if tuner == "xgb" or tuner == "xgb-rank":
            tuner_obj = XGBTuner(task, loss_type="rank")
        elif tuner == "ga":
            tuner_obj = GATuner(task, pop_size=50)
        elif tuner == "random":
            tuner_obj = RandomTuner(task)
        elif tuner == "gridsearch":
            tuner_obj = GridSearchTuner(task)
        elif tuner == "droplet":
            tuner_obj = DropletTuner(task)
        else:
            raise ValueError("Invalid tuner: " + tuner)

        n_trial = len(task.config_space)
        # do tuning
        start = time.time()
        tuner_obj.tune(
            n_trial=n_trial,
            early_stopping=early_stopping,
            measure_option=measure_option,
            callbacks=[
                #autotvm.callback.progress_bar(n_trial, prefix=prefix),
                autotvm.callback.log_to_file(log_filename_tmp),
            ],
        )
        end = time.time()

        best_avg, best_std, config = get_best_time(log_filename_tmp)
        print("Time tuning %s: %.4f, %.4f, %.2f " %(tuner, best_avg, best_std, end-start), config)
        #print(config)
        
        f = open(log_filename_tmp, "r")
        f1 = open(log_filename, "a")
        for l in f.readlines():
            f1.write(l)
            f1.close
        f1.close()
        f.close()
            
        
########################################################################
# Finally, we launch tuning jobs and evaluate the end-to-end performance.

def evaluate_performance(lib, data_shape):
    # load parameters
    dev = tvm.device(str(target), 0)
    module = runtime.GraphModule(lib["default"](dev))
    data_tvm = tvm.nd.array((np.random.uniform(size=data_shape)).astype(dtype),device=dev)
    module.set_input("data", data_tvm)

    print(module.benchmark(dev, number=1, repeat=600))


def tune_and_evaluate(tuning_opt, log_file):
    # extract workloads from relay program
    mod, params, data_shape, out_shape = get_network(model_name, batch_size)
    tasks = autotvm.task.extract_from_program(
        mod["main"], target=target, params=params, ops=(relay.op.get("nn.conv2d"),)
    )

    if os.path.exists(log_file):
        os.remove(log_file)

    # run tuning tasks
    tune_kernels(tasks, **tuning_opt)

    # compile kernels in kernel tuned only mode
    with autotvm.apply_history_best(log_file):
        with tvm.transform.PassContext(opt_level=3):
            lib = relay.build_module.build(mod, target=target, params=params)
        evaluate_performance(lib, data_shape)

# We do not run the tuning in our webpage server since it takes too long.
# Uncomment the following line to run it by yourself.
#tuner = ["droplet", "gridsearch", "random", "ga", "xgb"]

tuner = ["droplet"]

for t in tuner:

    log_file = "results/cuda_%s_%s.log" % (model_name, t)

    tuning_option = {
        "log_filename": log_file,
        "tuner": t,
        "early_stopping": 600,
        "measure_option": autotvm.measure_option(
            builder=autotvm.LocalBuilder(),
            runner=autotvm.LocalRunner(
                number=2, 
                repeat=5,
                timeout=4, 
                min_repeat_ms=150,
            ),
        ),
    }
    tune_and_evaluate(tuning_option, log_file)