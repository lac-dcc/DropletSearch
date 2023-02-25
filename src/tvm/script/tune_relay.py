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
from tvm import relay, autotvm
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

#################################################################
# Configure tensor tuning settings and create tasks
# -------------------------------------------------

# You can skip the implementation of this function for this tutorial.
def tune_kernels(
    tasks, measure_option, tuner="gridsearch", early_stopping=None, log_filename="tuning.log"
):
    total_time_tuning = 0
    for i, task in enumerate(tasks):
        log_filename_tmp = log_filename + "_layer_" + str(i) + ".log"

        if os.path.exists(log_filename_tmp):
            os.remove(log_filename_tmp)
        
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
        with tvm.transform.PassContext(opt_level=3):
            tuner_obj.tune(
                n_trial=n_trial,
                early_stopping=early_stopping,
                measure_option=measure_option,
                callbacks=[
                    #autotvm.callback.progress_bar(n_trial),
                    autotvm.callback.log_to_file(log_filename_tmp),
                ],
            )
        end = time.time()

        best_avg, best_std, config = get_best_time(log_filename_tmp)
        total_time_tuning += (end-start)
        print("Time partial tuning: %.4f, %.4f, %.2f" %(best_avg, best_std, end-start))
        #print(config)
        
        f = open(log_filename_tmp, "r")
        f1 = open(log_filename, "a")
        for l in f.readlines():
            f1.write(l)
        f1.close()
        f.close()
    print("Time total tuning: %.2f" %(total_time_tuning))
            
        
########################################################################
# Finally, we launch tuning jobs and evaluate the end-to-end performance.

def tune_and_evaluate(tuning_opt, log_file, model, arch, tuner, only_eval, target):
    # extract workloads from relay program
    mod, params, data_shape, out_shape = get_network(model, batch_size)
    tasks = autotvm.task.extract_from_program(
        mod["main"], target=target, params=params, ops=(relay.op.get("nn.conv2d"),)
    )

    print(only_eval)

    if only_eval != 1:
        if os.path.exists(log_file):
            os.remove(log_file)
        # run tuning tasks
        tune_kernels(tasks, **tuning_opt)

    print("without opt")
    lib = relay.build(mod, target=target, params=params)
    evaluate_performance(lib, data_shape, target)
    
    print("%s with opt" %(tuner))
    # compile kernels in kernel tuned only mode
    with autotvm.apply_history_best(log_file):
        with tvm.transform.PassContext(opt_level=3):
            lib = relay.build(mod, target=target, params=params)
            evaluate_performance(lib, data_shape, target)
    
if __name__ == "__main__":

    if len(sys.argv) > 4:
        model = sys.argv[1]
        tuner = sys.argv[2]
        arch = sys.argv[3]
        only_eval = sys.argv[4]
    else:
        print("Not valid configuration")
        exit()

    if "x86" in arch:
        target = "llvm"
    elif "cuda" in arch:
        target = "cuda"

    log_file = "results/%s/%s/%s/cpu.log" % (model, arch, tuner)

    tuning_option = {
        "log_filename": log_file,
        "tuner": tuner,
        "early_stopping": None,
        "measure_option": autotvm.measure_option(
            builder=autotvm.LocalBuilder(),
            runner=autotvm.LocalRunner(
                number=2, repeat=5, min_repeat_ms=100, enable_cpu_cache_flush=True if target=="llvm" else False
            ),
        ),
    }

    tune_and_evaluate(tuning_option, log_file, model, arch, tuner, only_eval, target)
