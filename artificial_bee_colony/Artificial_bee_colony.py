#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 12:34:00 2020

@author: mirauta
"""
import numpy as np

from scipy import optimize
from copy import deepcopy
from Artificial_bee_colony_objects import *
N=3
ND=1000
beta=np.random.uniform(size=N)
x0=np.random.normal(size=N*ND).reshape(N,-1)
y0=np.dot(abs(beta.reshape(1,N)),x0)

class ObjectiveFunction():

    def __init__(self, name, dim, minf, maxf):
        self.name = name
        self.dim = dim
        self.minf = minf
        self.maxf = maxf

    def sample(self):
        return np.random.uniform(low=self.minf, high=self.maxf, size=self.dim)

    def evaluate(self, beta):
        return ((y0-np.dot(beta.reshape(1,N),x0))**2).sum()
    
    def get_fitness(self,objective_value):       
        return 1 / (1 + objective_value) if objective_value >= 0 else 1 + np.abs(objective_value)

     
class SolutionFinder(object):

    def __init__(self, obj_function, colony_size=30, n_iter=5000, max_trials=100):

        self.colony_size = colony_size
        self.N_agents =int(self.colony_size /2)
        self.obj_function = obj_function

        self.n_iter = n_iter
        self.max_trials = max_trials

        self.optimal_solution = None
        self.optimality_tracking = []
        self.optimal_solution_tracking = []

    def __initialize_agents(self):
        self.solutions = [SolutionClass(self.obj_function)for itr in range(self.N_agents)]
        self.onlokeer_solutions = [SolutionClass(self.obj_function)for itr in range(self.N_agents)]
 

    def __explore_phase(self,solutions):
#        print ("explore phase")
        [solution.propose_merge_friend(Friendsolution=np.random.choice(self.solutions)) for solution in solutions]
        [solution.update_solution() for solution in solutions]


    def __get_solution_probabilities(self):

        sum_fitness = sum([solution.fitness for solution in self.solutions])
        for solution in self.solutions:solution.prob = solution.fitness / sum_fitness

    def __select_best_solutions(self,solutions):
        self.__get_solution_probabilities()
        
        
#        self.best_agents =  list(filter(lambda solution: solution.prob > np.random.uniform(low=0, high=1), agents))
        
        self.best_solutions =   np.hstack([np.repeat(solution, int(solution.prob*self.N_agents)) for solution in solutions])

    def __reset_phase(self):
        
        [solution.reset_solution(self.max_trials) for solution in self.solutions]

    def __update_optimal_solution(self):

        n_optimal_solution =  max(self.onlokeer_solutions + self.solutions,  key=lambda solution: solution.fitness)
        if not self.optimal_solution:
            self.optimal_solution = deepcopy(n_optimal_solution)
            
        else:
            if n_optimal_solution.fitness > self.optimal_solution.fitness:
                self.optimal_solution = deepcopy(n_optimal_solution)
        self.optimality_tracking.append(self.optimal_solution.fitness)
        self.optimal_solution_tracking.append(self.optimal_solution)

    def run(self):
#        self.__reset_algorithm()
        self.__initialize_agents()
        
        for itr in range(self.n_iter):

            self.__explore_phase(self.solutions)            
            
            self.__select_best_solutions(self.solutions)
#            print (list(map(lambda solution: solution.pos, self.best_solutions)))
            self.__explore_phase(self.best_solutions) 
#            self.__onlooker_solutions_phase()
#            self.__reset_phase()

            self.__update_optimal_solution()

#            print("iter: {} ".format(itr, "%04.03e"))
#            print (self.optimal_solution.pos)
#            print(self.solutions[0].fitness)
#            print(self.solutions[0].proposed_fitness)
            
#            print("iter: {} = cost: {}"
#                  .format(itr, "%04.03e" % self.optimal_solution.fitness))

abc = SolutionFinder(obj_function=ObjectiveFunction(name='Sphere', dim=N, minf=-10.0, maxf=10.0),
                        colony_size=8, n_iter=1000, max_trials=90)
abc.run()
#print (abc.optimal_solution.pos)
abc.obj_function.evaluate(np.array(beta))
abc.obj_function.evaluate(np.array(-beta))

print (beta)
print (np.vstack(list(map(lambda sol:sol.pos, abc.solutions))))
print (np.around(list(map(lambda sol:sol.prob, abc.solutions)),1))
print (list(map(lambda sol:sol.trial, abc.solutions)))

#print (list(map(lambda sol:sol.track_pos, abc.solutions[:1])))
#print (list(map(lambda sol:sol.pos, abc.optimal_solution_tracking)))

#print (list(map(lambda sol:sol.fitness, abc.optimal_solution_tracking)))
print(beta)

#print (list(map(lambda solution:solution.prob,abc.solutions)))
print ("END")
