#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 15:21:07 2023

@author: jefftamburi
"""

import numpy as np
import random


num_letters
seq_length
num_mutations
num_generations
max_children
max_children_list=[i for i in range(1, max_children+1)]


def mutate_sequence(sequence):
    index_to_mutate = random.randint(0, len(sequence) - 1)
    sequence[index_to_mutate]=random.randint(0,num_letters - 1)
    return sequence

root=np.random.choice(num_letters, seq_length)
num_nodes=np.zeros(num_generations) #number of nodes at each generation (from generation 0, root)
num_nodes[0]=1 
                
for gen in range(1,num_generations+1): #generation 0 is root (done), last generation does not have children
    for in range(np.random.choice(max_children_list)): #drawing how many children this 
        num_children[gen]=child
        for nm in range (num_mutations):
            mutate_sequence(sequence)
        seq_mat=
            
    
    