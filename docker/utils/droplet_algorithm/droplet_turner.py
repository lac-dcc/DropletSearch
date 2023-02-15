"""Tuner with genetic algorithm"""

import numpy as np
from scipy import stats
from .tuner import Tuner

class DropletTuner(Tuner):
    """Tuner with droplet algorithm.
    This tuner does not have a cost model so it always run measurement on real machines.
    This tuner expands the :code:`ConfigEntity` as gene.

    INPUTS: Task 
            start_position : [OPTIONAL] = position zero is default
    """

    def __init__(self, task, start_position=None):
        super(DropletTuner, self).__init__(task)
        
        self.max_value = 99999
        self.best_choice = []
        self.best_res = [self.max_value]

        # space info
        self.space = task.config_space
        self.dims = []
        self.next = []

        for k, v in self.space.space_map.items():
            self.dims.append(len(v))
        
        self.total_number_execution = int(np.mean(self.dims))
        self.next = [[0] * len(self.dims)] if start_position == None else start_position
        self.number_execution = 1
        self.best_choice = [-1] * len(self.dims)
        self.visited = self.next
    
    def create_space(self, value):
        b = str(0) * (len(self.dims) - len(bin(value)[2:])) + bin(value)[2:]
        return [int(x) for x in b]

    '''
        Return the search space 
    '''
    def generate_search_space(self):
        search_space = []
        for i in range(1,2**len(self.dims)):
            search_space.append([x for x in self.create_space(i)])
            search_space.append([-x for x in self.create_space(i)])
        return search_space
    
    def generate_new_positions(self):
        new_p = []
        if self.number_execution < self.total_number_execution:
            for p in self.generate_search_space():
                new_p.append([i * self.number_execution for i in p])
        return new_p
    
    def next_positions(self, new_positions):
        next_set = []
        for p in new_positions:
            p = [x + y for x, y in zip(p, self.best_choice)]
            if p not in self.visited and self.safe_space(p):
                next_set.append(p)
                self.visited.append(p)
        return next_set
    
    def safe_space(self, data):
        for d in data:
            if d < 0:
                return False
        return True

    '''
        Return the p_value between two arrays
    '''
    def p_value(self, elem_1, elem_2):
        data_1 = np.array(elem_1)
        data_2 = np.array(elem_2)
        if len(data_1) <= 1 or len(data_2) <= 1: # Case that there is only one element
            return 0
        return stats.ttest_ind(data_1, data_2).pvalue

    '''
        Return the next batch 
    '''
    def next_batch(self, batch_size):
        ret = []
        for value in self.next:
            index, exp = 0, 1
            for i in range(0, len(value)):
                index += (value[i] % self.dims[i]) * exp
                exp *= self.dims[i]
            ret.append(self.space.get(index))
        return ret
    
    '''
        Update the search space by measuring time 
    '''
    def update(self, inputs, results):

        found_best_pos = False
        for i, (inp, res) in enumerate(zip(inputs, results)):
            try:
                y = np.mean(res.costs)
                if np.mean(self.best_res) > y and self.p_value(self.best_res, res.costs) <= 0.05:
                    self.best_res = res.costs
                    self.best_choice = self.next[i]
                    found_best_pos = True
            except:
                y = self.max_value

        self.next = []
        if not found_best_pos:
            self.next = self.next_positions(self.generate_new_positions())
            self.number_execution += 1
        else:
            self.next = self.next_positions(self.generate_search_space())

        #print("Best", self.best_choice, "n_exec", self.number_execution)
        #print("next", self.next)
            

    '''
        Check for search space
    '''
    def has_next(self):
        return self.number_execution < self.total_number_execution or len(self.next) > 0

    def load_history(self, data_set, min_seed_records=500):
        pass