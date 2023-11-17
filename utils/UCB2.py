import numpy as np
            
import torch
import torch.nn.functional as F

import matplotlib
import matplotlib.pyplot as plt
import networkx as nx

# from utils.graph_utils import *
            
class NodeUCBTS():
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
            self.children_values = {i : -1000*self.probs[i, node_idx] for i in self.next_possible_children} # we build the values for all of the next possible children
            self.children = {i : None for i in self.next_possible_children}
            
    def choose_next_children(self):
        max_key = min(self.children_values, key=lambda cle: self.children_values[cle])
        return max_key
       
    def get_children(self, children_idx):
        if self.children[children_idx] == None:
            next_possible_children = [i for i in self.next_possible_children if i!=children_idx]
            next_lenght = self.actual_lenght + self.lengths[self.node_idx, children_idx].item()
            next_lenght = self.actual_lenght * self.probs[self.node_idx, children_idx]
            self.children[children_idx] = NodeUCBTS(next_possible_children, self, children_idx, self.probs, self.c, self.lengths, next_lenght, self.depth + 1, self.root) 
        return self.children[children_idx]
    
    def compute_temperature(self, tau):
        if self.father != None : 
            self.temp_prob = self.probs[self.node_idx, self.father.node_idx]**(1/tau) / sum(self.probs[:, self.father.node_idx]**(1/tau))
    
    def update_values(self, lenght_path):
        if self.father != None:
            # self.UCB_value = np.sqrt(self.father.n_exploration) / (1 + self.n_exploration) * self.probs[self.node_idx, self.father.node_idx]
            self.UCB_value = np.sqrt(self.father.n_exploration) / (1 + self.n_exploration) * self.temp_prob
            self.average_value = (self.average_value * self.n_exploration + lenght_path)/(self.n_exploration+1)
            if self.root.max_lenght - self.root.min_lenght != 0:
                self.node_value = (self.average_value - self.root.min_lenght)/(self.root.max_lenght - self.root.min_lenght) - self.UCB_value 
            else : 
                self.node_value = (self.average_value - self.root.min_lenght)/(self.root.max_lenght) - self.UCB_value 
            self.n_exploration += 1

            self.father.children_values[self.node_idx] = self.node_value
            self.father.update_values(lenght_path)
        else : 
            # print(lenght_path)
            pass
                       
class UCBTreeSearch():
    def __init__(self, probs, lengths, c , start_idx = 0): # probs is an n x n np array st probs[i,j] = proba qu il y ait un lien entre les deux 
        num_nodes, _ = probs.shape
        self.lengths = lengths
        self.num_nodes = num_nodes
        self.c = c / (2*num_nodes)
        self.start_idx = start_idx
        self.next_possible_states = [i for i in range(num_nodes) if i != start_idx]
        self.children_values = {i : probs[start_idx, i] for i in self.next_possible_states} # we build the values for all of the next possible children
        actual_lenght = 1
        self.min_lenght = -1
        self.max_lenght = 0
        self.root = NodeUCBTS(self.next_possible_states, None, start_idx, probs, c/num_nodes, lengths, actual_lenght, 1, self)

    def one_search(self):
        next_children = self.root.choose_next_children()
        node = self.root.get_children(next_children)
        lenght_path = self.lengths[self.start_idx, node.node_idx].item()
        for _ in range(self.num_nodes - 2):
            next_children = node.choose_next_children()
            node_idx = node.node_idx
            node = node.get_children(next_children)
            lenght_path += self.lengths[node_idx, node.node_idx].item()
        lenght_path += self.lengths[self.start_idx, node.node_idx].item()
        self.update_min_max(lenght_path, node)
        node.update_values(lenght_path)
    
    def update_min_max(self, lenght, node):
        if self.min_lenght > lenght or self.min_lenght == -1:
            self.min_node = node
            self.min_lenght = lenght
        if self.max_lenght < lenght:
            self.max_lenght = lenght
        
    def search(self, max_trials):
        for _ in range(max_trials):
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
    
def UCBSearch_with_batch_UCB2(probs, edges, max_trials= 10_000, c=1):
    batch_size = probs.shape[0]
    TSP_return = torch.zeros_like(edges)
    for i in range(batch_size):
        print('begin')
        print(edges)
        TS = UCBTreeSearch(probs[i], edges[i], c)
        TSP_appro_i = TS.return_TSP_approx(max_trials)
        TSP_return[i] = torch.tensor(TSP_appro_i)
        print('end')
        print(edges)
    return TSP_return
        


if __name__ == "__main'__":
    probs = np.eye(4)
    lengths = np.ones_like(probs)

    n = 10  # Remplacez 5 par la valeur de votre choix pour la taille du tableau n x n.

    # Générer les indices i et j
    i, j = np.meshgrid(np.arange(n), np.arange(n))

    # Calculer le tableau des longueurs
    lengths = np.abs(i - j)
    probs = np.exp(lengths)

    probs = 1/probs

    for i in range(n):
        probs[i, i] = 0

    # print(probs)
    # print(lengths)

    UCB = UCBTreeSearch(probs, lengths, c=1)
    # min_node, min_lenght_path = UCB.search(100)
    # min_lenght = 0
    # node = min_node
    # while node.father != None:
    #     print(node.node_idx, " <- ", end='')
    #     min_lenght += lengths[node.node_idx, node.father.node_idx]
    #     node = node.father
    # min_lenght += lengths[node.node_idx, min_node.node_idx]
    # print(node.node_idx, " <- ", end='')
    # print("min_lenght : ",min_lenght)
    # print(UCB.min_lenght)

    # min_lenght = 0
    # node = UCB.min_node
    # while node.father != None:
    #     print(node.node_idx, " <- ", end='')
    #     min_lenght += lengths[node.node_idx, node.father.node_idx]
    #     node = node.father
    # min_lenght += lengths[node.node_idx, min_node.node_idx]
    # print(node.node_idx, " <- ", end='')
    # print("min_lenght : ",min_lenght)
    UCB.return_TSP_approx(1)


    # Idees : random start pour ajouter un peu de random