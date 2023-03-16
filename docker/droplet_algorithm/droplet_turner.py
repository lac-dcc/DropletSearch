"""Tuner with droplet algorithm"""

import numpy as np
from scipy import stats
from .tuner import Tuner

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
        self.dims = []

        for k, v in self.space.space_map.items():
            self.dims.append(len(v))
        
        # start position
        start_position =  [0] * len(self.dims) if start_position == None else start_position
        self.next = [(self.convert_idx(start_position),start_position)]
        self.visited = {}
        self.visited[self.convert_idx(start_position)] = 1
        self.batch, self.count, self.max_value, self.pvalue = 0, 0, 99999, pvalue
        self.best_choice = (-1, [-1] * len(self.dims), [self.max_value])
        # number execution is important when the start position is not valid
        self.number_execution = 2
        self.total_number_execution = max(self.dims)
    
    def number_to_bin(self, value, factor=1):
        """ convert a number to a binary vector.
        """
        bin_format = str(0) * (len(self.dims) - len(bin(value)[2:])) + bin(value)[2:]
        return [int(i) * factor for i in bin_format]

    def convert_idx(self, value):
        #print(value)
        index, exp = 0, 1
        for i in range(0, len(value)):
            index += (value[i] % self.dims[i]) * exp
            exp *= self.dims[i]
        return index

    def new_search_space(self, factor=1):
        '''Return the new search space 
        '''
        search_space = []
        for i in range(0,len(self.dims)): # [0,0,0] => [0,0,1], [0,1,0], [1,0,0]
            search_space.append(self.number_to_bin(2**i, factor))
            search_space.append(self.number_to_bin(2**i, -factor))
        return search_space
    
    def safe_value(self, data):
        for d in data:
            if d < 0:
                return False
        return True

    def next_positions(self, new_positions):
        next_set = []
        for p in new_positions:
            new_p = [x + y for x, y in zip(p, self.best_choice[1])]
            idx_p = self.convert_idx(new_p)
            if self.safe_value(new_p) and idx_p not in self.visited.keys():
                self.visited[idx_p] = 1
                next_set.append((idx_p,new_p))
        return next_set

    def p_value(self, elem_1 : np.array, elem_2 : np.array):
        '''Return the p_value between two arrays
        '''
        if len(elem_1) <= 1 or len(elem_2) <= 1: # Case that there is only one element
            return True
        return stats.ttest_ind(elem_1, elem_2).pvalue <= self.pvalue

    def next_batch(self, batch_size):
        '''Return the next batch 
        '''
        ret = []
        self.count, self.batch = 0, batch_size
        for i in range(batch_size):
            if i >= len(self.next):
                break
            self.count += 1
            ret.append(self.space.get(self.next[i][0]))
        return ret

    def update_next_element(self):
        return self.next[self.count:-1]
    
    def update(self, inputs, results):
        '''Update the search space by measuring time 
        '''
        found_best_pos = False
        for i, (inp, res) in enumerate(zip(inputs, results)):
            try:
                if np.mean(self.best_choice[2]) > np.mean(res.costs) and self.p_value(np.array(self.best_choice[2]), np.array(res.costs)):
                    self.best_choice = (self.next[i][0], self.next[i][1], res.costs)
                    found_best_pos = True
            except:
                continue

        self.next = self.update_next_element()
        while len(self.next) < self.batch and self.number_execution < self.total_number_execution:
            if found_best_pos:
                self.next += self.next_positions(self.new_search_space())
                found_best_pos = False
            else:
                self.next += self.next_positions(self.new_search_space(self.number_execution))
                self.number_execution += 1

    def has_next(self):
        return len(self.next) > 0

    def load_history(self, data_set, min_seed_records=500):
        pass