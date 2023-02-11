"""Tuner with genetic algorithm"""

import numpy as np
from scipy import stats

from .tuner import Tuner
from .model_based_tuner import knob2point, point2knob

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
        self.keys = []
        self.dims = []
        self.next = []
        self.start_pos = 1

        for k, v in self.space.space_map.items():
            self.keys.append(k)
            self.dims.append(len(v))
        
        self.next = [[0] * len(self.dims)] if start_position == None else start_position
        self.number_execution = 1

        self.best_choice = [-1] * len(self.keys)
        self.visited = self.next
    
    def create_space(self, value):
        b = str(0) * (len(self.dims) - len(bin(value)[2:])) + bin(value)[2:]
        return [int(x) for x in b]

    '''
        Return the search space 
    '''
    def generate_search_space(self):
        search_space = []
        for i in range(1,2**len(self.keys)):
            p = [x for x in self.create_space(i)]
            p_inv = [-x for x in self.create_space(i)]
            search_space.append(p)
            search_space.append(p_inv)
        return search_space
    
    def generate_new_positions(self):
        new_p = []
        if self.number_execution < min(self.dims):
            for p in self.generate_search_space():
                new_p.append([i * self.number_execution for i in p])
        return new_p
    
    def update_new_positions(self, new_positions):
        next_set = []
        for p in new_positions:
            print(p, end=" => ")
            p = [x + y for x, y in zip(p, self.best_choice)]
            print(p)
            if p not in self.visited and self.safe_space(p):
                self.visited.append(p)
                next_set.append(p)
        return next_set
    
    def safe_space(self, data):
        for i, d in enumerate(data):
            if d < 0 or d >= self.dims[i]:
                return False
        return True

    '''
        Return the p_value between two arrays
    '''
    def p_value(self, elem_1, elem_2):
        data_1 = np.array(elem_1)
        data_2 = np.array(elem_2)
        if len(data_1) <= 1 or len(data_2): # Case that there is only one element
            return 0
        return stats.ttest_ind(data_1, data_2).pvalue

    '''
        Return the next batch 
    '''
    def next_batch(self, batch_size):
        ret = []
        print(self.next)
        for value in self.next:
            index, exp = 0, 1
            for i in range(0, len(value)):
                index += value[i] * exp
                exp *= self.dims[i]
            if index >= 0:
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
            except:
                y = self.max_value
            if np.mean(self.best_res) > y and self.p_value(self.best_res, res.costs) <= 0.5:
                self.best_res = res.costs
                self.best_choice = self.next[i]
                found_best_pos = True

        self.next = []
        if self.best_res[0] == self.max_value:
            self.start_pos += 1
            self.next = self.create_space(self.start_pos)
        elif not found_best_pos:
            # Update new positions
            new_positions = self.generate_search_space()
            # Update the next positions
            self.next = self.update_new_positions(new_positions)
            self.number_execution += 1
        else:
            # Update new positions
            new_positions = self.generate_new_positions()
            # Update the next positions
            self.next = self.update_new_positions(new_positions)

        print("Best", self.best_choice, "n_exec", self.number_execution)
        print("next", self.next)
            

    '''
        Check for search space
    '''
    def has_next(self):
        return len(self.next) > 0 

    def load_history(self, data_set, min_seed_records=500):
        pass