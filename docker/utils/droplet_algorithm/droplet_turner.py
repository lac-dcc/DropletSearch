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
            trying : [OPTIONAL] = 1 is default  
    """

    def __init__(self, task, start_position=None, trying=1):
        super(DropletTuner, self).__init__(task)
        
        self.best_choice = []
        self.best_res = [99999]

        # space info
        self.space = task.config_space
        self.keys = []
        self.dims = []
        self.next = []
        self.trying = trying
        self.number_execution = 1

        for k, v in self.space.space_map.items():
            self.keys.append(k)
            self.dims.append(len(v))
        
        self.position = self.generate_search_space(len(self.keys)) 
        self.next = [[0] * len(self.dims)] + self.position if start_position == None else start_position + self.position

        self.best_choice = [-1] * len(self.keys)
        self.visited = self.next
    
    '''
        Return the search space 
    '''
    def generate_search_space(self, size):
        space_search = []
        for i in range(1,2**size):
            b = str(0) * (size - len(bin(i)[2:])) + bin(i)[2:]
            v = []
            for k in b:
                v.append(int(k))
            space_search.append(v)
        return space_search
    
    def safe_space(self, data):
        for i, d in enumerate(data):
            if d >= self.dims[i]:
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
        for value in self.next:
            index = 0
            exp = 1
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

        #print("tentativa: ", self.trying)
        found_best_pos = False
        for i, (inp, res) in enumerate(zip(inputs, results)):
            try:
                y = np.mean(res.costs)
            except:
                y = 99999
            if np.mean(self.best_res) > y and self.p_value(self.best_res, res.costs) <= 0.5:
                self.best_res = res.costs
                self.best_choice = self.next[i]
                found_best_pos = True

        if found_best_pos or self.trying > 0:
            # Update new positions
            if self.number_execution < min(self.dims):
                for p in self.position.copy():
                    new_p = [i * self.number_execution for i in p]
                    self.position.append(new_p)
            
            # Update the next positions
            next_set = []
            for p in self.position:
                p = [x + y for x, y in zip(p, self.best_choice)]
                if p not in self.visited and self.safe_space(p):
                    self.visited.append(p)
                    next_set.append(p)
            self.next = next_set.copy()

            if not found_best_pos:
                self.trying -= 1
        else:
            self.next = []

    '''
        Check for search space
    '''
    def has_next(self):
        self.number_execution += 1
        return len(self.next) > 0 

    def load_history(self, data_set, min_seed_records=500):
        pass