import numpy as np
            
import torch
import torch.nn.functional as F

import matplotlib
import matplotlib.pyplot as plt
import networkx as nx

from tqdm import tqdm 

import random

# from utils.graph_utils import *
            
class NodeBeamSearch():
    def __init__(self, next_possible_children, father, node_idx, probs, c, lengths, actual_lenght, depth, root, tau = 1):
        self.c = c
        self.root = root
        self.lengths = lengths
        self.actual_lenght = actual_lenght
        self.father = father
        self.probs = probs
        self.depth = depth
        self.n_exploration = 0
        self.average_value = 0
        self.node_idx = node_idx
        self.next_possible_children = next_possible_children
        self.compute_temperature(tau)
        if len(next_possible_children) == 0:
            self.children_values = None
            self.children = None
        else : 
            self.next_proba = self.compute_next_children_proba()
            self.children_values = {i : self.probs[i, node_idx] for i in self.next_possible_children} # we build the values for all of the next possible children
            self.children = {i : None for i in self.next_possible_children}
            
    def choose_next_children(self):
        random_children = np.random.choice(len(self.next_proba), p=self.next_proba)
        return random_children
    
    def compute_next_children_proba(self):
        n_nodes, _ = self.probs.shape
        next_proba = np.zeros(n_nodes)
        for i in self.next_possible_children:
            next_proba[i] = self.probs[i, self.node_idx]
        sum_proba = np.sum(next_proba)
        if sum_proba != 0:
            next_proba /= sum_proba
            return next_proba
        for i in self.next_possible_children:
            next_proba[i] = 1/n_nodes
        return next_proba
            
    def get_children(self, children_idx):
        if self.children[children_idx] == None:
            next_possible_children = [i for i in self.next_possible_children if i!=children_idx]
            next_lenght = self.actual_lenght + self.lengths[self.node_idx, children_idx].item()
            next_lenght = self.actual_lenght * self.probs[self.node_idx, children_idx]
            self.children[children_idx] = NodeBeamSearch(next_possible_children, self, children_idx, self.probs, self.c, self.lengths, next_lenght, self.depth + 1, self.root) 
        return self.children[children_idx]
    
    def compute_temperature(self, tau):
        if self.father != None : 
            self.temp_prob = self.probs[self.node_idx, self.father.node_idx]**(1/tau) / sum(self.probs[:, self.father.node_idx]**(1/tau))

                       
class BeamSearch():
    def __init__(self, probs, lengths, c ): # probs is an n x n np array st probs[i,j] = proba qu il y ait un lien entre les deux 
        num_nodes, _ = probs.shape
        self.num_nodes = num_nodes
        self.lengths = lengths
        self.num_nodes = num_nodes
        self.c = c / (2*num_nodes)
        self.roots = []
        actual_lenght = 1
        self.min_lenght = -1
        self.max_lenght = 0
        for idx in range(num_nodes):
            next_possible_states = [i for i in range(num_nodes) if i != idx]
            self.children_values = {i : probs[idx, i] for i in next_possible_states} # we build the values for all of the next possible children
            self.roots.append(NodeBeamSearch(next_possible_states, None, idx, probs, c/num_nodes, lengths, actual_lenght, 1, self))   
    def choose_root(self):
        return self.roots[random.randint(0, self.num_nodes-1)]
    
    def one_search(self):
        node = self.choose_root()
        root_idx = node.node_idx
        lenght_path = 0
        for _ in range(self.num_nodes - 1):
            next_children = node.choose_next_children()
            node_idx = node.node_idx
            node = node.get_children(next_children)
            lenght_path += self.lengths[node_idx, node.node_idx].item()
        lenght_path += self.lengths[root_idx, node.node_idx].item()
        self.update_min_max(lenght_path, node)
        # node.update_values(lenght_path)
    
    def update_min_max(self, lenght, node):
        if self.min_lenght > lenght or self.min_lenght == -1:
            self.min_node = node
            self.min_lenght = lenght
        if self.max_lenght < lenght:
            self.max_lenght = lenght
        
    def search(self, max_trials):
        for _ in tqdm(range(max_trials)):
            self.one_search()
        return self.min_node, self.min_lenght
            
                
    def return_TSP_approx(self, max_trials):
        min_node, min_lenght = self.search(max_trials)
        tsp_approx = np.zeros_like(min_node.probs)
        node = min_node
        while node.father != None:
            tsp_approx[node.node_idx, node.father.node_idx] = 1
            tsp_approx[node.father.node_idx, node.node_idx] = 1
            node = node.father
        tsp_approx[node.node_idx, min_node.node_idx] = 1
        tsp_approx[min_node.node_idx, node.node_idx] = 1
        return tsp_approx
    
def beamSearch_with_batch02(probs, edges, max_trials= 10_000, c=1):
    batch_size = probs.shape[0]
    TSP_return = torch.zeros_like(edges)
    for i in range(batch_size):
        print("Batch Beam Search number : ", i,' on ', batch_size)
        TS = BeamSearch(probs[i], edges[i], c)
        TSP_appro_i = TS.return_TSP_approx(max_trials)
        TSP_return[i] = torch.tensor(TSP_appro_i)
    return TSP_return