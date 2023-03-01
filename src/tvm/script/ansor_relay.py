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
Auto-scheduling a Neural Network for x86 CPU
============================================
**Author**: `Lianmin Zheng <https://github.com/merrymercy>`_, \
            `Chengfan Jia <https://github.com/jcf94/>`_

Auto-tuning for specific devices and workloads is critical for getting the
best performance. This is a tutorial on how to tune a whole neural
network for CPU and GPU with the auto-scheduler.
"""

import numpy as np

import tvm
from tvm import relay, auto_scheduler
from tvm.relay import data_dep_optimization as ddo
from utils import get_network, get_best_time, evaluate_performance
import tvm.relay.testing
import tvm.contrib.graph_executor as runtime
import time, sys, os

#################################################################
# Define a Network
# ----------------
use_sparse = False
batch_size = 1
dtype = "float32"
input_name = "data"
num_threads = os.cpu_count()
os.environ["TVM_NUM_THREADS"] = str(num_threads)

#################################################################
# Extract Search Tasks

def run_tuning(model, arch, log_file, only_eval, target):

    mod, params, input_shape, output_shape = get_network(model, batch_size, "NCHW", dtype=dtype)
    #print("Extract tasks...")
    if only_eval != 1:

        if os.path.exists(log_file):
            os.remove(log_file)

        tasks, task_weights = auto_scheduler.extract_tasks(mod["main"], params, target)

        tuner = auto_scheduler.TaskScheduler(tasks, task_weights)
        tune_option = auto_scheduler.TuningOptions(
            num_measure_trials=10000,  # change this to 20000 to achieve the best performance
            runner=auto_scheduler.LocalRunner(number=2, repeat=3, min_repeat_ms=100, enable_cpu_cache_flush=True if target=="llvm" else False),
            measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
            verbose=0
        )
        start = time.time()
        tuner.tune(tune_option)
        end = time.time()
        print("Time search %.4f" %(end-start))

    print("ansor with opt")
    with auto_scheduler.ApplyHistoryBest(log_file):
        with tvm.transform.PassContext(opt_level=3, config={"relay.backend.use_auto_scheduler": True}):
            lib = relay.build(mod, target=target, params=params)
            evaluate_performance(lib, input_shape, target, "data")


if __name__ == "__main__":

    model = sys.argv[1]
    arch = sys.argv[2]
    only_eval = sys.argv[3]

    if "x86" in arch:
        target = "llvm"
    elif "cuda" in arch:
        target = "cuda"

    log_file = "results/%s/%s/ansor/cpu.log" % (model, arch)

    run_tuning(model, arch, log_file, only_eval, target)
