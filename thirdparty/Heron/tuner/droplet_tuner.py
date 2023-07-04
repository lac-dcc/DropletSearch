import os
import time
from .tuner import Tuner
import numpy as np

class DropletTuner(Tuner):
    def optimize(self, env, pop, stat, s_time):
        print(self.config)
        samples = [] + pop
        for i in range(self.config.iter_walks):
            population = self.constrained_random_sample(env, self.config.pop_num)
            print(self.config.pop_num)
            samples = samples + population
            perfs = [x.predict for x in samples]
            stat.append([np.array(perfs).max(), time.time() - s_time])
        return population, samples
    
    def walk(self):
        pass



        


