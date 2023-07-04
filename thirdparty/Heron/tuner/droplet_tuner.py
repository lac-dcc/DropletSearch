import os
import time
from .tuner import Tuner
from Heron.sample import *
from Heron.utils import *
import numpy as np

class DropletTuner(Tuner):
    def optimize(self, env, pop, stat, s_time):
        samples = [] + pop
        print(len(pop))
        for i in range(self.config.iter_walks):
            population = self.walk(env, self.config.pop_num)
            samples = samples + population
            perfs = [x.predict for x in samples]
            stat.append([np.array(perfs).max(), time.time() - s_time])
        return population, samples

    def constrained_random_sample_sequential(self, task, number, config):
        ret = []
        for i in range(number):
            sample = Sample(task)
            valid, point = sample.knob_manager.randSample({})
            sample.point = point
            sample.valid = valid
            code = Code(point)
            #print(valid, point, code)
            sample.stmt_code = code
            ret.append(sample)
        return ret

    def walk(self, env, number, timeout=10):
        samples = self.constrained_random_sample_sequential(env.task, number, self.config)
        for sample in samples:
            sample.predict = self.cost_model.predict([sample])[0]
            sample.violation = 0
        return samples



        



