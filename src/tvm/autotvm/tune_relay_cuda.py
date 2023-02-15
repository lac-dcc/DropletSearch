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
Auto-tuning a Convolutional Network for NVIDIA GPU
==================================================
**Author**: `Lianmin Zheng <https://github.com/merrymercy>`_, `Eddie Yan <https://github.com/eqy/>`_

Auto-tuning for specific devices and workloads is critical for getting the
best performance. This is a tutorial on how to tune a whole convolutional
network for NVIDIA GPU.

The operator implementation for NVIDIA GPU in TVM is written in template form.
The template has many tunable knobs (tile factor, unrolling, etc).
We will tune all convolution and depthwise convolution operators
in the neural network. After tuning, we produce a log file which stores
the best knob values for all required operators. When the TVM compiler compiles
these operators, it will query this log file to get the best knob values.

We also released pre-tuned parameters for some NVIDIA GPUs. You can go to
`NVIDIA GPU Benchmark <https://github.com/apache/tvm/wiki/Benchmark#nvidia-gpu>`_
to see the results.

Note that this tutorial will not run on Windows or recent versions of macOS. To
get it to run, you will need to wrap the body of this tutorial in a :code:`if
__name__ == "__main__":` block.
"""

######################################################################
# Install dependencies
# --------------------
# To use the autotvm package in tvm, we need to install some extra dependencies.
# (change "3" to "2" if you use python2):
#
# .. code-block:: bash
#
#   pip3 install --user psutil xgboost tornado cloudpickle
#
# To make TVM run faster during tuning, it is recommended to use cython
# as FFI of tvm. In the root directory of tvm, execute:
#
# .. code-block:: bash
#
#   pip3 install --user cython
#   sudo make cython3
#
# Now return to python code. Import packages.

# sphinx_gallery_start_ignore
# sphinx_gallery_requires_cuda = True
# sphinx_gallery_end_ignore
import os
import time

import numpy as np

import tvm
from tvm import relay, autotvm
import tvm.relay.testing
from tvm.autotvm.tuner import XGBTuner, GATuner, RandomTuner, GridSearchTuner, DropletTuner
import tvm.contrib.graph_executor as runtime

import json

#################################################################
# Define Network
# --------------
# First we need to define the network in relay frontend API.
# We can load some pre-defined network from :code:`tvm.relay.testing`.
# We can also load models from MXNet, ONNX and TensorFlow.

#### DEVICE CONFIG ####
target = tvm.target.cuda()

#### TUNING OPTION ####
network = "resnet-18"
dtype = "float32"

def get_best_time(log, ms=True):

    f = open(log, "r")
    best_avg = 9999.0
    best_std = 0.0
    for line in f.readlines():
        data = json.loads(line)
        r = np.mean(data["result"][0])
        if (best_avg > r):
            best_avg = r
            best_std = np.std(data["result"][0])
    f.close()

    if ms: # convet to ms
        best_avg *= 1000
        best_std *= 1000
    return best_avg, best_std

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
        mod, params = relay.frontend.from_mxnet(block, shape={"data": input_shape}, dtype=dtype)
        net = mod["main"]
        net = relay.Function(
            net.params, relay.nn.softmax(net.body), None, net.type_params, net.attrs
        )
        mod = tvm.IRModule.from_expr(net)
    else:
        raise ValueError("Unsupported network: " + name)

    return mod, params, input_shape, output_shape


###########################################
# Set Tuning Options
# ------------------
# Before tuning, we apply some configurations.



####################################################################
#
# .. note:: How to set tuning options
#
#   In general, the default value provided here works well.
#
#   If you have large time budget, you can set :code:`n_trial`, :code:`early_stopping` larger,
#   which makes the tuning runs longer.
#
#   If you have multiple devices, you can use all of them for measurement to
#   accelerate the tuning process. (see the 'Scale up measurement` section below).
#

###################################################################
# Begin Tuning
# ------------
# Now we can extract tuning tasks from the network and begin tuning.
# Here, we provide a simple utility function to tune a list of tasks.
# This function is just an initial implementation which tunes them in sequential order.
# We will introduce a more sophisticated tuning scheduler in the future.

# You can skip the implementation of this function for this tutorial.
def tune_tasks(
    tasks,
    measure_option,
    tuner="xgb",
    n_trial=1000,
    early_stopping=None,
    log_filename="tuning.log",
):
    # create tmp log file
    if os.path.exists(log_filename):
        os.remove(log_filename)

    for i, tsk in enumerate(reversed(tasks)):
        prefix = "[Task %2d/%2d] " % (i + 1, len(tasks))

        # create tuner
        if tuner == "xgb" or tuner == "xgb-rank":
            tuner_obj = XGBTuner(tsk, loss_type="rank")
        elif tuner == "ga":
            tuner_obj = GATuner(tsk, pop_size=100)
        elif tuner == "random":
            tuner_obj = RandomTuner(tsk)
        elif tuner == "gridsearch":
            tuner_obj = GridSearchTuner(tsk)
        elif tuner == "droplet":
            tuner_obj = DropletTuner(tsk)
        else:
            raise ValueError("Invalid tuner: " + tuner)

        # do tuning
        tsk_trial = min(n_trial, len(tsk.config_space))
        tuner_obj.tune(
            n_trial=tsk_trial,
            early_stopping=early_stopping,
            measure_option=measure_option,
            callbacks=[
                #autotvm.callback.progress_bar(tsk_trial, prefix=prefix),
                autotvm.callback.log_to_file(log_filename),
            ],
        )


########################################################################
# Finally, we launch tuning jobs and evaluate the end-to-end performance.


def tune_and_evaluate(tuning_opt, tuner, log_file):
    # extract workloads from relay program
    #print("Extract tasks...")
    mod, params, input_shape, out_shape = get_network(network, batch_size=1)
    tasks = autotvm.task.extract_from_program(
        mod["main"], target=target, params=params, ops=(relay.op.get("nn.conv2d"),)
    )

    # run tuning tasks
    start = time.time()
    tune_tasks(tasks, **tuning_opt)
    end = time.time()

    best_avg, best_std = get_best_time(log_file)
    print("Time tuning %s: %.4f, %.4f, %.2f" %(tuner, best_avg, best_std, end-start))


# We do not run the tuning in our webpage server since it takes too long.
# Uncomment the following line to run it by yourself.

tuner = ["droplet", "xgb", "ga", "random", "gridsearch"]

for t in tuner:

    log_file = "results/gpu_%s_%s.log" % (network, t)

    tuning_option = {
        "log_filename": log_file,
        "tuner": t,
        "n_trial": 170,
        "early_stopping": None,
        "measure_option": autotvm.measure_option(
            builder=autotvm.LocalBuilder(),
            runner=autotvm.LocalRunner(number=2, repeat=5, timeout=4),
        ),
    }

    tune_and_evaluate(tuning_option, t, log_file)
