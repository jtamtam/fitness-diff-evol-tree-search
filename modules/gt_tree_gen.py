import math
import jax.numpy as jnp
from jax import vmap
from jax import random
from jax import jit
from functools import partial

import h5py
import numpy as np

from typing import Dict, List
from jaxtyping import Array, Float

#Nico file imports
import numpy as np
import random as rdm

from math import exp, expm1
from datetime import datetime, date

import h5py
import os

from tqdm import tqdm


target_depth = 0
num_mutations = 0
sequences = []
n_letters = 0
###
#Nico's functions

startTime = datetime.now()
today = date.today()


def LireH5(path):
    file = h5py.File(path, 'r')

    sequences_leaves = file['Chains']

    Jij  = np.array(file['Matrix_contact'])
    # nbrspin = 200
    
    mutations = np.array(file['Mutations'])
    sequences_tree = np.array(file['allchains'])

    sequences_leaves = np.array(sequences_leaves)
    sequences_tree = np.array(sequences_tree)
    file.close()
    return sequences_leaves,sequences_tree,mutations,Jij

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
            #generate rdm number uniformly between 0 and 1, to compare it
            #with p.
            randnumber = rdm.uniform(0,1)
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
        rand_site1 = rdm.randrange(0,len(chain),1)
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

            if rdm.uniform(0,1) < exp((energy-new_energ)/Temp):

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



###

def mutate(key, exclude_indexes):
  '''
    Mutates a sequence by randomly changing one of the letters

    Accepts:
      exclude_indexes: list of indexes to exclude from the mutation (shape = (n_mutations,1))

    Returns:
      mutated sequence
  '''
  global n_letters

  space = jnp.tile(jnp.arange(n_letters), (num_mutations, 1)) 
  options  = space[space != exclude_indexes].reshape((space.shape[0],space.shape[1] - exclude_indexes.shape[1]))

  key_ = random.split(key, num_mutations)

  mutations = vmap(random.choice)(key_, options)

  return jnp.expand_dims(mutations, axis=1)

def create_seq(key, seq, depth):

  k1, k2 = random.split(key)
  
  if(depth > target_depth):
    return

  random_indexes = [[] for x in range(2)]
  
  random_indexes[0]= random.choice(k1, jnp.array(range(len(seq))), (num_mutations,1), replace=False)
  random_indexes[1]= random.choice(k2, jnp.array(range(len(seq))), (num_mutations,1), replace=False)


  seq_1 = seq
  seq_2 = seq

  seq_1 = seq_1.at[random_indexes[0]].set(mutate(k1, seq[random_indexes[0]]))
  seq_2 = seq_2.at[random_indexes[1]].set(mutate(k2, seq[random_indexes[1]]))


  #seq
  sequences[depth].append(seq_1)
  create_seq(k1, seq_1, depth+1)
  
  #new_seq
  sequences[depth].append(seq_2)
  create_seq(k2, seq_2, depth+1)


def generate_groundtruth(metadata : Dict[str, int], seed = 42, verbose = False) -> List[Array]: 
    '''
        Generates a groundtruth example based on the metadata provided


        Args : 
            metadata: dictionary containing the required specifications
            seed: random seed
            verbose: print the number of leaves generated

        Returns:
            masked_main : masked sequences (shape = (n_all, seq_length))
            true_main   : true sequences   (shape = (n_all, seq_length))
            tree        : tree structure   (shape = (n_all, n_all))
    '''

    # key = random.PRNGKey(seed)

    # global target_depth
    global num_mutations
    # global sequences
    # global n_letters

    num_mutations = metadata['n_mutations']
    n_all       = metadata['n_all']
    n_leaves    = metadata['n_leaves']
    n_ancestors = metadata['n_ancestors']
    seq_length  = metadata['seq_length']
    n_letters   = metadata['n_letters']

    target_depth = int(math.log2(n_leaves))-1

    # seq = jnp.zeros(seq_length, dtype=jnp.int64)

    # sequences = [[] for x in range(target_depth + 1)]
    # create_seq(key,seq,0)

    # if(verbose):
    #   print(len(sequences[target_depth]))

    # ### copy the leaves
    # masked_main = sequences[target_depth].copy()
    # true_main   = sequences[target_depth].copy()

    # n_leaves    = len(masked_main)
    n_ancestors = n_leaves - 1


    # for i in range(0,n_ancestors):
    #     masked_main.append(seq)

    # ## Generate the true sequences by traversing through the leaves to the root
    # for i in range(len(sequences)-2, -1, -1):  #commence à -2 car déjà copié les leaves
    #     for k in range(0,len(sequences[i])):
    #         true_main.append(sequences[i][k])
    # true_main.append(seq)

    # if(verbose):
    #   i = 0
    #   for item in true_main:
    #     print(">seq"+str(i))
    #     for char in item:
    #       print(char, end="")
    #     print()

    #     i+=1
    
    ###NEW    
    # sequences_leaves, sequences_tree, mutations, Jij=LireH5("mutations_1_2_temperature_5_generations_5_length_200/mutations_1_2_temperature_5_generations_5_length__200_seed2_nbrfile0.h5")        
    
    # new_seq_leaves=jnp.array(sequences_leaves[0]).astype(jnp.bfloat16)
    # true_main=jnp.array(sequences_tree[0]).astype(jnp.bfloat16)
    # masked_main=jnp.array(sequences_tree[0]).astype(jnp.bfloat16)
    
    # for i in range(len(new_seq_leaves[:,]),len(true_main[:,])): 
    #     masked_main=masked_main.at[i].set(jnp.zeros(len(new_seq_leaves[1,:]),dtype=jnp.bfloat16))
    ###
    
    ###Integrating Nico's code
    
    ## number of mutations m (here we do trees with m =1, then m = 2 until m = 50)
    mutations_list = list(np.linspace(num_mutations,num_mutations,1, dtype = int)) #here 50 mutations 
    Temperature = 5
    
    #number of generations in the tree
    number_generations = int(math.log2(n_leaves))
    
    ##length of protein
    number_spins = seq_length
    
    ##seed 
    seed = 42 #just random seed
    rdm.seed(seed)
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
    
    for idxm, mut in enumerate(mutations_list):
    
            date = today.strftime("%Y_%m_%d_")
            hour = startTime.strftime('%H_%M_%S_')
            
            number_mutations = mut
            
            
            Tree = Generate_tree(number_generations)
            
            ##ancestor sequence
            starting_chain = np.zeros(number_spins, dtype = np.int8)
            
            
            ### change (now -1, 1 states) -> 0-19
            for k in range(0,number_spins):
                     starting_chain[k] = rdm.randrange(0,2,1)
                    
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
    
    new_seq_leaves=jnp.array(final_chains[0]).astype(jnp.bfloat16)
    true_main=jnp.array(all_chains[0]).astype(jnp.bfloat16)
    masked_main=jnp.array(all_chains[0]).astype(jnp.bfloat16)
    
    #emptying masked_main to use as base of Gradient Descent
    for i in range(len(new_seq_leaves[:,]),len(true_main[:,])): 
        masked_main=masked_main.at[i].set(jnp.zeros(len(new_seq_leaves[1,:]),dtype=jnp.bfloat16))
    
    ###
    
    
    
    tree = []
    
    iter = n_ancestors

    for i in range(0,len(masked_main)):
        leave = [0]*len(masked_main)

        if(i%2==0 and i!=len(masked_main)-1):
            iter +=1 

        leave[iter] = 1
        tree.append(leave)

    print(
        #"type of masked_main:", masked_main.type,'\n'
        "length of masked_main:", len(masked_main),'\n','\n'
        #"type of true_main:", true_main.type,'\n'
        "length of true_main:", len(true_main),'\n','\n'
        #"type of tree:", tree.type,'\n'
        "length of tree:", len(tree),'\n'
          )

    return jnp.array(masked_main).astype(jnp.bfloat16), jnp.array(true_main).astype(jnp.bfloat16), jnp.array(tree).astype(jnp.bfloat16)
