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
        self.dims = []

        for _, v in self.space.space_map.items():
            self.dims.append(len(v))
        
        #print(self.dims)
        # start position
        start_position =  [0] * len(self.dims) if start_position == None else start_position
        self.next = [(knob2point(start_position, self.dims),start_position)]
        self.visited = set([knob2point(start_position, self.dims)])
        self.batch, self.count, self.pvalue = 16, 0, pvalue
        self.best_choice = (-1, [-1] * len(self.dims), [99999])
        # number execution is important when the start position is not valid
        self.number_execution = 2
        self.total_number_execution = max(self.dims)
    
    def number_to_bin(self, value, factor=1):
        """ convert a number to a binary vector.
        """
        bin_format = str(0) * (len(self.dims) - len(bin(value)[2:])) + bin(value)[2:]
        return [int(i) * factor for i in bin_format]

    def get_search_space(self, factor=1):
        '''Return the new search space 
        '''
        search_space = []
        for i in range(1,2**len(self.dims)): # [0,0,0] => [0,0,1], [0,1,0], [1,0,0]. ...
            search_space.append(self.number_to_bin(i, factor))
            search_space.append(self.number_to_bin(i, -factor))
        return search_space

    def next_positions(self, new_positions):
        next_set = []
        for p in new_positions:
            new_p = [(x + y) % self.dims[i] if x + y > 0 else 0 for i, (x, y) in enumerate(zip(p, self.best_choice[1]))]
            idx_p = knob2point(new_p, self.dims)
            if idx_p not in self.visited: # memoization
                self.visited.add(idx_p)
                next_set.append((idx_p,new_p))
        return next_set

    def p_value(self, elem_1 : np.array, elem_2 : np.array):
        '''Return the p_value between two arrays, using Student's t-test
        '''
        return True if len(elem_1) <= 1 or len(elem_2) <= 1 else stats.ttest_ind(elem_1, elem_2).pvalue <= self.pvalue

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
        
        #print("exec:", self.number_execution)
        #print("best:", self.best_choice)
        self.next = self.next[self.count:-1]

        if found_best_pos:
            self.next += self.next_positions(self.get_search_space())

            # Prediction the gradient descendent direction and fill the search space
            while len(self.next) < self.batch:
                self.next += self.next_positions(self.get_search_space(self.number_execution))
                self.number_execution += 1

        #while len(self.next) < self.batch and self.number_execution < self.total_number_execution:
        #    if found_best_pos:
        #        self.next += self.next_positions(self.local_search_space())
        #        found_best_pos = False
        #    else:
        #        self.next += self.next_positions(self.new_search_space(self.number_execution))
        #        self.number_execution += 1
        #        break
        #print("next_elements:", self.next)

    def has_next(self):
        return len(self.next) > 0