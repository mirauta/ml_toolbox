#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 12:34:00 2020

@author: mirauta
"""
import numpy as np

from scipy import optimize
from copy import deepcopy
        

class SolutionClass(object):


    def __init__(self, obj_function):
        self.obj_function = obj_function
        self.minf = obj_function.minf
        self.maxf = obj_function.maxf
        self.trial = 0
        self.prob = 0
        self.test=0

        self.pos = obj_function.sample()
        self.track_pos=[] 
        self.objective = obj_function.evaluate(self.pos)
        self.fitness =self.obj_function.get_fitness(self.objective)

        self.proposed=None
        self.proposed_fitness=0
        
        
    def evaluate_boundaries(self, pos):
        if (pos < self.minf).any() or (pos > self.maxf).any():
            pos[pos > self.maxf] = self.maxf
            pos[pos < self.minf] = self.minf
        return pos



    def reset_solution(self, max_trials):

        if self.trial >= max_trials:
            self.pos = self.obj_function.sample()
            self.fitness = self.obj_function.evaluate(self.pos)
            self.trial = 0
            self.prob = 0
            print("reset solution")


    def propose_merge_friend (self,  Friendsolution,max_trials=10):
#        print("explore solution")
        if self.trial >= max_trials: return
        phi = np.random.uniform(low=-1, high=1, size=len(self.pos))
        self.proposed=self.pos+(self.pos - Friendsolution.pos) * phi
        self.proposed = self.evaluate_boundaries(self.proposed)
        
    def update_solution(self):
#        print("update solution")
    
        proposed_objective = self.obj_function.evaluate(self.proposed)
        self.proposed_fitness =self.obj_function.get_fitness(proposed_objective)

        if self.proposed_fitness > self.fitness:
            self.pos = self.proposed
            self.fitness = self.proposed_fitness
            self.trial = 0
           
        else:
            self.trial += 1      
        self.track_pos.append(self.pos)
