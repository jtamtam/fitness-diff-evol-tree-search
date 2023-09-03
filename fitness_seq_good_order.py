#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 15:16:07 2020

Phylogeny

"""
import numpy as np
import random

from tqdm import tqdm
from math import exp, expm1
from datetime import datetime, date

import h5py
import os

from tqdm import tqdm

startTime = datetime.now()
today = date.today()


def Generate_Graph(Nspin, p):

    #generate a matrix of size len(N)xlen(N), which will return the coupled
    #pairs. The coupled pairs are represented by the indices of the matrix,
    #the entry of the matrix for the coresponding pair is 1 for coupled pairs or
    #0 if they are not linked.
    mat = np.zeros([Nspin, Nspin])

    #iterate over the upper diagonal (diagonal not included)
    #i.e. we iterate over the pairs of the chain.
    for i in range(0,Nspin):
        for j in range(i+1, Nspin):
            #generate random number uniformly between 0 and 1, to compare it
            #with p.
            randnumber = random.uniform(0,1)
            #if the random number is above p than accept the link, i.e. matrix
            #entry is set to 1, otherwise it is set to 0.
            if randnumber >= p:
                mat[i,j] = 1
            else:
                mat[i,j] = 0
    # return the final matrix with the couplings.
    return mat




def Mutate(chain, max_nbr_mut, Temp, matcontact):
    count_mut = 0
    energy = (-1)*np.matmul(chain,np.matmul(matcontact,chain))
    mutation_list = np.zeros(max_nbr_mut)
    # print('chain before mutate', chain)
    while(count_mut < max_nbr_mut):
        newchain = chain.copy()
        rand_site1 = random.randrange(0,len(chain),1)
        newchain[rand_site1] = newchain[rand_site1]*-1
        new_energ = (-1)*np.matmul(newchain,np.matmul(matcontact,newchain))
        # print('new proposed chain', newchain)
        # print('energy', energy)
        # print('new energy', new_energ)
        if new_energ < energy:
            # print('energy is lowered')
            #if new config has lower energy, than the new config is accepted
            #and the new values are set.
            
            energy = new_energ
            chain = newchain.copy()
            mutation_list[count_mut] = rand_site1
            count_mut = count_mut + 1

        
        else:
            # print('new config has higher energy')
            #if the new configuration has higher energy than generate a random
            #number uniformly between 0 and 1 and compute the exponential of the
            #energy difference.

            #if the randomly chosen number is lower than the probability given by
            #the exp than accept the new configuration.

            if random.uniform(0,1) < exp((energy-new_energ)/Temp):

                energy = new_energ
                chain = newchain.copy()
                mutation_list[count_mut] = rand_site1

                count_mut = count_mut + 1

            

    return chain,mutation_list

def Generate_tree(nbr_gen):
    tree = {}
    for g in range(0,nbr_gen+1):
        list_children = np.linspace(1,pow(2,g),pow(2,g), dtype = np.int16)
        for child in list(list_children):
            tree['{}/{}'.format(g,child)] = None
    return tree

##number of realisations
number_files = 1
for nbrf in tqdm(range(0,number_files)):
    ## number of mutations m (here we do trees with m =1, then m = 2 until m = 50)
    mutations_list = list(np.linspace(50,50,1, dtype = int)) #here 50 mutations 
    Temperature = 5
    
    #number of generations in the tree
    number_generations = 5
    
    ##length of protein
    number_spins = 200
    
    ##seed 
    seed = 2*(nbrf+1)
    random.seed(seed)
    Jij =  np.load('./Jij.npy')
    # proba = 0.02
    # nbravailablestartingeqchains = 10000
    
    ##saves sequences at the leaves (last generation)
    final_chains = np.zeros((len(mutations_list),pow(2,number_generations),number_spins),dtype = np.int8)
    
    ##save sequences of all tree
    save_chainsandmutations = True
    if save_chainsandmutations:
        all_chains = np.zeros((len(mutations_list),2*pow(2,number_generations)-1,number_spins))  
        # savedmutations_list = np.zeros((len(mutations_list),2*pow(2,number_generations)-2, number_mutations))
        
    print('ATTENTION TO STARTING CHAIN IF EQUILIBRIUM CHAIN IS USED')
    
    for idxm, mut in enumerate(mutations_list):
    
            date = today.strftime("%Y_%m_%d_")
            hour = startTime.strftime('%H_%M_%S_')
            
            number_mutations = mut
            
            
            Tree = Generate_tree(number_generations)
            
            ##ancestor sequence
            starting_chain = np.zeros(number_spins, dtype = np.int8)
            
            
            ### change (now -1, 1 states) -> 0-19
            for k in range(0,number_spins):
                     starting_chain[k] = random.randrange(0,2,1)
                    
            # OPEN sequence at equilibrium
            #starting_chain = ocd.OneClusterChainp0_02N200(Temperature, indexstartingchain)

            Tree['0/1'] = starting_chain
            ct = 2*pow(2,number_generations)-1-1
            if save_chainsandmutations:
                all_chains[idxm,ct,:] = starting_chain
                #print(ct)
                
            for g in range(1,number_generations+1):
                
                list_parents = np.linspace(1,pow(2,g-1),pow(2,g-1), dtype = np.int16)

                ctprime = ct - pow(2,g)-1
                ct = ct-pow(2,g)
                for parent in list_parents:

                    chain = Tree['{}/{}'.format(g-1,parent)]
                    newchain1,ml1 = Mutate(chain.copy(), number_mutations, Temperature, Jij)
                    newchain2,ml2 = Mutate(chain.copy(), number_mutations, Temperature, Jij)
        
                    ###save all sequences along tree

                    if save_chainsandmutations:
                        all_chains[idxm,ctprime+1,:] = newchain1
                        all_chains[idxm,ctprime+2,:] = newchain2
                        # print(ctprime+1)
                        # print(ctprime+2)
                        ctprime = ctprime+2
                    
                    Tree['{}/{}'.format(g,2*parent-1)] = newchain1
                    Tree['{}/{}'.format(g,2*parent)] = newchain2

    
            for index_chain, child in enumerate(list(np.linspace(1,pow(2,number_generations),pow(2,number_generations),dtype = np.int16))):
                final_chains[idxm,index_chain,:] = Tree['{}/{}'.format(number_generations,child)]
            
            
            
            
    ##########SAVE CHAINS, CREATES FOLDER IF NOT EXISTING FOR THE PARAMETERS MUTATIONS/GENERATIONS#######################
    
    folderpath =  './mutations_{}_{}_temperature_{}_generations_{}_length_{}/'.format(int(min(mutations_list)),int(max(mutations_list)),int(Temperature),int(number_generations),int(number_spins))
    if not os.path.exists(folderpath):
        os.makedirs(folderpath)
        
    
    
    filename = folderpath + 'mutations_{}_{}_temperature_{}_generations_{}_length__{}_seed{}_nbrfile{}.h5'.format(int(min(mutations_list)),int(max(mutations_list)),int(Temperature),int(number_generations),int(number_spins),int(seed),nbrf)
    
    file = h5py.File(filename, 'w')
    file.close()
    
    para = [number_spins, Temperature, number_generations,seed]
    
    file = h5py.File(filename, 'r+')
    file.create_dataset('Parameters', data = np.array(para))
    file.create_dataset('Matrix_contact', data = Jij)
    file.create_dataset('Chains', data = final_chains, compression='gzip', compression_opts=9)
    file.create_dataset('Mutations', data = np.array(mutations_list))
    if save_chainsandmutations:
        file.create_dataset('allchains', data = all_chains, compression='gzip', compression_opts=9)
        # file.create_dataset('mutations', data = mutations_list, compression='gzip', compression_opts=9)
    file.close()
