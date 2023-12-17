"""Tuner with droplet algorithm"""

import numpy as np
from scipy import stats
from .tuner import Tuner
from .model_based_tuner import knob2point

class DropletTuner(Tuner):
    """Tuner with droplet algorithm.
    This tuner does not have a cost model so it always run measurement on real machines.
    This tuner expands the :code:`ConfigEntity` as gene.

    INPUTS: Task 
            start_position : [OPTIONAL] = default is [0,0,...,0]
            pvalue : [OPTIONAL] = default is 0.05
    """

    def __init__(self, task, start_position=None, pvalue=0.05):
        super(DropletTuner, self).__init__(task)

        # space info
        self.space = task.config_space
        self.dims, self.iter = [], 0

        for _, v in self.space.space_map.items():
            self.dims.append(len(v))
        
        # start position
        start_position =  [0] * len(self.dims) if start_position == None else start_position
        self.best_choice = (-1, [0] * len(self.dims), [99999])
        self.visited = set([knob2point(start_position, self.dims)])
        self.execution, self.total_execution, self.batch = 1, max(self.dims), 16
        self.count, self.pvalue, self.step = 0, pvalue, max(1, self.total_execution//20)
        self.next = [(knob2point(start_position, self.dims),start_position)] + self.speculation()
    
    def num_to_bin(self, value, factor=1):
        """ convert a number to a binary vector.
        """
        bin_format = str(0) * (len(self.dims) - len(bin(value)[2:])) + bin(value)[2:]
        return [int(i) * factor for i in bin_format]

    def search_space(self, factor=1):
        search_space = []
        for i in range(2**len(self.dims)-1,0,-1): # [0,0,0] => [0,0,1], [0,1,0], [1,0,0]. ...
            search_space += [self.num_to_bin(i, factor)] + [self.num_to_bin(i, -factor)]
        return search_space

    def next_pos(self, new_positions):
        next_set = []
        for p in new_positions:
            if len(next_set) > self.batch:
                break
            new_p = [(x+y) % self.dims[i] if (x+y>0) else 0 for i, (x, y) in enumerate(zip(p, self.best_choice[1]))]
            idx_p = knob2point(new_p, self.dims)
            if idx_p not in self.visited: # memoization
                self.visited.add(idx_p)
                next_set.append((idx_p,new_p))
        return next_set

    def p_value(self, elem_1 : np.array, elem_2 : np.array):
        '''Return the p_value between two arrays, using Student's t-test'''
        return True if len(elem_1) <= 1 or len(elem_2) <= 1 else stats.ttest_ind(elem_1, elem_2).pvalue <= self.pvalue

    def next_batch(self, batch_size):
        '''Return the next batch'''
        ret, self.batch = [], batch_size
        for i in range(batch_size):
            if i >= len(self.next):
                break
            ret.append(self.space.get(self.next[i][0]))
        return ret

    def speculation(self):
        # Gradient descending direction prediction and search space filling
        new_points = []
        while len(new_points) < self.batch and self.execution < self.total_execution:
            self.execution += self.step
            new_points += self.next_pos(self.search_space(self.execution))
        return new_points
    
    def update(self, inputs, results):
        found_best_pos = False
        count_valids = 0
        for i, (inp, res) in enumerate(zip(inputs, results)):
            try:
                if np.mean(self.best_choice[2]) > np.mean(res.costs) and self.p_value(np.array(self.best_choice[2]), np.array(res.costs)):
                    self.best_choice = (self.next[i][0], self.next[i][1], res.costs)
                    found_best_pos = True
                count_valids += 1
            except:
                continue

        self.next = self.next[self.batch:-1]
        if found_best_pos:
            self.next += self.next_pos(self.search_space())
            self.execution = 1
        self.next += self.speculation() 
        self.iter += 1
        if count_valids == 0 and self.iter > 3:
            self.next = []
            print(f"Warning: early termination due to an all-invalid neighborhood after {self.iter} iterations")

    def has_next(self):
        return len(self.next) > 0 