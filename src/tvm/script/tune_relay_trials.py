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

Auto-tuning a Convolutional Network for x86 CPU
===============================================
**Author**: `Yao Wang <https://github.com/kevinthesun>`_, `Eddie Yan <https://github.com/eqy>`_
"""

import os
import numpy as np
import time, sys

import tvm
from tvm import relay, autotvm, auto_scheduler
from tvm.relay import testing
from tvm.autotvm.tuner import XGBTuner, GATuner, RandomTuner, GridSearchTuner, DropletTuner
from tvm.autotvm.graph_tuner import DPTuner, PBQPTuner
from utils import get_network, get_best_time, evaluate_performance
import tvm.contrib.graph_executor as runtime

#################################################################
# Define network
# --------------
batch_size = 1
dtype = "float32"
input_name = "data"
num_threads = os.cpu_count()
os.environ["TVM_NUM_THREADS"] = str(num_threads)
        
########################################################################
# Finally, we launch tuning jobs and evaluate the end-to-end performance.

def tune_and_evaluate(tuning_opt, log_file_original, log_file_save, model, arch, tuner, target, trials, pvalue):
    # extract workloads from relay program
    mod, params, data_shape, out_shape = get_network(model, batch_size)
    
    if os.path.exists(log_file_save):
        os.remove(log_file_save)

    print("With opt")
    if tuner == "ansor":
        f = open(log_file_original, "r")
        f1 = open(log_file_save, "w")
        count_line = 0
        for l in f.readlines():
            if count_line > trials:
                break
            f1.write(l)
            count_line += 1
        f1.close()
        f.close()

        tasks, task_weights = auto_scheduler.extract_tasks(mod["main"], params, target)
        
        # compile kernels in kernel tuned only mode
        with auto_scheduler.ApplyHistoryBest(log_file_save):
            with tvm.transform.PassContext(opt_level=3, config={"relay.backend.use_auto_scheduler": True}):
                lib = relay.build(mod, target=target, params=params)
                evaluate_performance(lib, data_shape, target)

    else:
        tasks = autotvm.task.extract_from_program(mod["main"], target=target, params=params, ops=(relay.op.get("nn.conv2d"),))
        # run tuning tasks
        partial_trial = trials // len(tasks)
        for i, task in enumerate(tasks):
            log_filename_tmp = log_file_original + "_layer_" + str(i) + ".log"

            f = open(log_filename_tmp, "r")
            f1 = open(log_file_save, "a")
            count_line = 0
            for l in f.readlines():
                if count_line > partial_trial:
                    break
                f1.write(l)
                count_line += 1
            f1.close()
            f.close()

        # compile kernels in kernel tuned only mode
        with autotvm.apply_history_best(log_file_save):
            with tvm.transform.PassContext(opt_level=3):
                lib = relay.build(mod, target=target, params=params)
                evaluate_performance(lib, data_shape, target)
    os.remove(log_file_save)
    
    #print("without opt")
    #lib = relay.build(mod, target=target, params=params)
    #evaluate_performance(lib, data_shape, target)
    
    
if __name__ == "__main__":

    pvalue = 0.05
    if len(sys.argv) > 4:
        model = sys.argv[1]
        tuner = sys.argv[2]
        arch = sys.argv[3]
        trials = int(sys.argv[4])
    else:
        print("Not valid configuration")
        exit()

    if "x86" in arch:
        target = "llvm"
    elif "cuda" in arch:
        target = "cuda"
    elif "arm" in arch:
        target = "llvm -device=arm_cpu"

    log_file_original = "results/%s/%s/%s/cpu.log" % (arch, model, tuner)
    log_file_save = "results/%s/%s/%s/cpu_%d.log" % (arch, model, tuner, trials)

    tuning_option = {
        "log_filename": log_file_original,
        "tuner": tuner,
        "early_stopping": None,
        "measure_option": autotvm.measure_option(
            builder=autotvm.LocalBuilder(),
            runner=autotvm.LocalRunner(
                number=2, repeat=5, min_repeat_ms=100, enable_cpu_cache_flush=True if target=="llvm" else False
            ),
        ),
    }

    tune_and_evaluate(tuning_option, log_file_original, log_file_save, model, arch, tuner, target, trials, pvalue)
