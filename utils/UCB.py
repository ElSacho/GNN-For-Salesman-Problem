import numpy as np
            
import torch
import torch.nn.functional as F

import matplotlib
import matplotlib.pyplot as plt
import networkx as nx

from utils.graph_utils import *
            
class NodeUCBTS():
    def __init__(self, next_possible_children, father, node_idx, probs, c, lengths, actual_lenght, depth):
        self.c = c
        self.lengths = lengths
        self.actual_lenght = actual_lenght
        self.father = father
        self.probs = probs
        self.depth = depth
        self.n_exploration = 1
        self.UCB_value = self.c * np.sqrt(2*np.log(self.probs.shape[0] + 1)/self.n_exploration) + (1/self.probs.shape[0])**self.depth / self.n_exploration  #TODO : change the trai number
        self.node_idx = node_idx
        self.next_possible_children = next_possible_children
        if len(next_possible_children) == 0:
            self.children_values = None
            self.children = None
        else : 
            self.children_values = {i : probs[node_idx, i] for i in self.next_possible_children} # we build the values for all of the next possible children
            self.children = {i : None for i in self.next_possible_children}
            
    def choose_next_children(self):
        max_key = max(self.children_values, key=lambda cle: self.children_values[cle])
        return max_key
       
    def get_children(self, children_idx):
        if self.children[children_idx] == None:
            next_possible_children = [i for i in self.next_possible_children if i!=children_idx]
            next_lenght = self.actual_lenght + self.lengths[self.node_idx, children_idx]
            next_lenght = self.actual_lenght * self.probs[self.node_idx, children_idx]
            self.children[children_idx] = NodeUCBTS(next_possible_children, self, children_idx, self.probs, self.c, self.lengths, next_lenght, self.depth + 1) 
        return self.children[children_idx]
    
    def update_values(self, lenght_path, trial_number):
        # print(self.node_idx, " <- ", end='')
        n = np.sqrt(trial_number)
        self.UCB_value += self.c * np.sqrt(2*np.log(n + 1)/self.n_exploration) 
        self.UCB_value = self.c * np.sqrt(2*np.log(self.probs.shape[0] + 1)/self.n_exploration) * (1/self.probs.shape[0])**len(self.next_possible_children)   #TODO : change the trai number
        self.UCB_value = self.n_exploration * (1/self.probs.shape[0])**len(self.next_possible_children)
        # print(self.UCB_value)
        exploration_value = lenght_path # TODO
        if self.father != None:
            exploration_value = self.probs[self.node_idx, self.father.node_idx]    
        self.node_value = exploration_value - self.UCB_value 
        self.n_exploration += 1
        if self.father != None:
            self.father.children_values[self.node_idx] = self.node_value
            self.father.update_values(lenght_path, trial_number)
            # print(self.node_idx, " -> ", end='')
        else : 
            # print(lenght_path)
            pass
                       
class UCBTreeSearch():
    def __init__(self, probs, lengths, c , start_idx = 0): # probs is an n x n np array st probs[i,j] = proba qu il y ait un lien entre les deux 
        num_nodes, _ = probs.shape
        self.num_nodes = num_nodes
        self.c = c / (2*num_nodes)
        self.start_idx = start_idx
        self.next_possible_states = [i for i in range(num_nodes) if i != start_idx]
        self.children_values = {i : probs[start_idx, i] for i in self.next_possible_states} # we build the values for all of the next possible children
        actual_lenght = 1
        self.root = NodeUCBTS(self.next_possible_states, None, start_idx, probs, c/num_nodes, lengths, actual_lenght, 1)

    def one_search(self, trial_number):
        next_children = self.root.choose_next_children()
        node = self.root.get_children(next_children)
        for _ in range(self.num_nodes - 2):
            next_children = node.choose_next_children()
            node = node.get_children(next_children)
        lenght_path = node.actual_lenght
        lenght_path *= node.probs[self.start_idx, node.node_idx]
        node.update_values(lenght_path, trial_number)
        return node, lenght_path
        
    def search(self, max_trials):
        min_lenght_path = -1
        for trial in range(max_trials):
            node, lenght_path = self.one_search(trial+1)
            if lenght_path > min_lenght_path or min_lenght_path == -1:
                min_lenght_path = lenght_path
                min_node = node
            # if min_node.n_exploration > 10:
            #     # print("potential break in : ", trial)
            #     # break
                # pass
        return min_node, min_lenght_path
                
    def return_TSP_approx(self, max_trials):
        min_node, _ = self.search(max_trials)
        tsp_approx = np.zeros_like(min_node.probs)
        node = min_node
        while node.father != None:
            # min_lenght += lengths[node.node_idx, node.father.node_idx]
            tsp_approx[node.node_idx, node.father.node_idx] = 1
            tsp_approx[node.father.node_idx, node.node_idx] = 1
            node = node.father
        tsp_approx[node.node_idx, min_node.node_idx] = 1
        tsp_approx[min_node.node_idx, node.node_idx] = 1
        # min_lenght += lengths[node.node_idx, min_node.node_idx]
        return tsp_approx
    
def UCBSearch_with_batch(probs, edges, max_trials= 10_000, c=1):
    batch_size = probs.shape[0]
    TSP_return = torch.zeros_like(edges)
    for i in range(batch_size):
        TS = UCBTreeSearch(probs[i], edges[i], c)
        TSP_appro_i = TS.return_TSP_approx(max_trials)
        TSP_return[i] = torch.tensor(TSP_appro_i)
    return TSP_return
        
    

def plot_tsp(p, x_coord, W, W_val, W_target, title="default"):
    """
    Helper function to plot TSP tours.
    
    Args:
        p: Matplotlib figure/subplot
        x_coord: Coordinates of nodes
        W: Edge adjacency matrix
        W_val: Edge values (distance) matrix
        W_target: One-hot matrix with 1s on groundtruth/predicted edges
        title: Title of figure/subplot
    
    Returns:
        p: Updated figure/subplot
    
    """

    def _edges_to_node_pairs(W):
        """Helper function to convert edge matrix into pairs of adjacent nodes.
        """
        pairs = []
        for r in range(len(W)):
            for c in range(len(W)):
                if W[r][c] == 1:
                    pairs.append((r, c))
        return pairs
    
    # G = nx.from_numpy_array(W_val) LEGACY
    G = nx.from_numpy_array(W_val)
    pos = dict(zip(range(len(x_coord)), x_coord.tolist()))
    adj_pairs = _edges_to_node_pairs(W)
    target_pairs = _edges_to_node_pairs(W_target)
    colors = ['g'] + ['b'] * (len(x_coord) - 1)  # Green for 0th node, blue for others
    nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=50)
    nx.draw_networkx_edges(G, pos, edgelist=adj_pairs, alpha=0.3, width=0.5)
    nx.draw_networkx_edges(G, pos, edgelist=target_pairs, alpha=1, width=1, edge_color='r')
    p.set_title(title)
    return p


def plot_tsp_heatmap(p, x_coord, W_val, W_pred, title="default"):
    """
    Helper function to plot predicted TSP tours with edge strength denoting confidence of prediction.
    
    Args:
        p: Matplotlib figure/subplot
        x_coord: Coordinates of nodes
        W_val: Edge values (distance) matrix
        W_pred: Edge predictions matrix
        title: Title of figure/subplot
    
    Returns:
        p: Updated figure/subplot
    
    """

    def _edges_to_node_pairs(W):
        """Helper function to convert edge matrix into pairs of adjacent nodes.
        """
        pairs = []
        edge_preds = []
        for r in range(len(W)):
            for c in range(len(W)):
                if W[r][c] > 0.25:
                    pairs.append((r, c))
                    edge_preds.append(W[r][c])
        return pairs, edge_preds
        
    G = nx.from_numpy_array(W_val)
    pos = dict(zip(range(len(x_coord)), x_coord.tolist()))
    node_pairs, edge_color = _edges_to_node_pairs(W_pred)
    node_color = ['g'] + ['b'] * (len(x_coord) - 1)  # Green for 0th node, blue for others
    nx.draw_networkx_nodes(G, pos, node_color=node_color, node_size=50)
    nx.draw_networkx_edges(G, pos, edgelist=node_pairs, edge_color=edge_color, edge_cmap=plt.cm.Reds, width=0.75)
    p.set_title(title)
    return p

    
def plot_predictions_UCBsearch(x_nodes_coord, x_edges, x_edges_values, y_edges, y_pred_edges, UCB_nodes, num_plots=3):
    """
    Plots groundtruth TSP tour vs. predicted tours (with beamsearch).
    
    Args:
        x_nodes_coord: Input node coordinates (batch_size, num_nodes, node_dim)
        x_edges: Input edge adjacency matrix (batch_size, num_nodes, num_nodes)
        x_edges_values: Input edge distance matrix (batch_size, num_nodes, num_nodes)
        y_edges: Groundtruth labels for edges (batch_size, num_nodes, num_nodes)
        y_pred_edges: Predictions for edges (batch_size, num_nodes, num_nodes)
        bs_nodes: Predicted node ordering in TSP tours after beamsearch (batch_size, num_nodes)
        num_plots: Number of figures to plot
    
    """
    y = F.softmax(y_pred_edges, dim=3)  # B x V x V x voc_edges
    y_bins = y.argmax(dim=3)  # Binary predictions: B x V x V
    y_probs = y[:,:,:,1]  # Prediction probabilities: B x V x V
    for f_idx, idx in enumerate(np.random.choice(len(y), num_plots, replace=False)):
        f = plt.figure(f_idx, figsize=(15, 5))
        x_coord = x_nodes_coord[idx].cpu().numpy()
        W = x_edges[idx].cpu().numpy()
        W_val = x_edges_values[idx].cpu().numpy()
        W_target = y_edges[idx].cpu().numpy()
        W_sol_bins = y_bins[idx].cpu().numpy()
        W_sol_probs = y_probs[idx].cpu().numpy()
        # W_bs = tour_nodes_to_W(UCB_nodes[idx].cpu().numpy())
        W_bs = UCB_nodes[idx].cpu().numpy()
        plt1 = f.add_subplot(131)
        plot_tsp(plt1, x_coord, W, W_val, W_target, 'Groundtruth: {:.3f}'.format(W_to_tour_len(W_target, W_val)))
        plt2 = f.add_subplot(132)
        plot_tsp_heatmap(plt2, x_coord, W_val, W_sol_probs, 'Prediction Heatmap')
        plt3 = f.add_subplot(133)
        plot_tsp(plt3, x_coord, W, W_val, W_bs, 'UCB Search: {:.3f}'.format(W_to_tour_len(W_bs, W_val)))
        plt.show()

                    

# if __name__ == "__main'__":
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

print(probs)
print(lengths)

UCB = UCBTreeSearch(probs, lengths, c=1)
min_node, min_lenght_path = UCB.search(100)
min_lenght = 0
node = min_node
while node.father != None:
    print(node.node_idx, " <- ", end='')
    min_lenght += lengths[node.node_idx, node.father.node_idx]
    node = node.father
min_lenght += lengths[node.node_idx, min_node.node_idx]
print(node.node_idx, " <- ", end='')
print("min_lenght : ",min_lenght)
print(UCB.return_TSP_approx(100))


# Idees : random start pour ajouter un peu de random