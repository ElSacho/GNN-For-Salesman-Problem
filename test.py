import networkx as nx
import numpy as np
import torch
import torch.nn as nn



# hidden_dim = 16
# number_of_nodes = 10
# batch_size = 32


# nodes_features = torch.rand(batch_size, number_of_nodes, hidden_dim)
# W1 = nn.Linear(hidden_dim, hidden_dim, False)
# nodes_features = W1(nodes_features)
# print(nodes_features.shape)

# edges = torch.randn(batch_size, number_of_nodes, number_of_nodes, hidden_dim)

# # Calcul de eta en utilisant la formule
# sigmoid_edges = torch.sigmoid(edges)  # Appliquer sigmoid à tous les éléments de edges
# sum_sigmoid_edges = sigmoid_edges.sum(dim=2, keepdim=True)  # Somme le long de la dimension j
# eta = sigmoid_edges / (sum_sigmoid_edges + 101)

# print(sum_sigmoid_edges.shape)


# Exemple de tensors et de modèle
# batch_size = 2
# N_nodes = 3
# H = 4

# # Créez des tensors de forme (batch_size, N_nodes, N_nodes, H) et (batch_size, N_nodes, H) pour eta et nodes_features
# eta = torch.randn(batch_size, N_nodes, N_nodes, H)
# nodes_features = torch.randn(batch_size, N_nodes, H)

# # Créez une couche linéaire W2
# W2 = nn.Linear(H, H, False)

# # Appliquez W2 à nodes_features
# nodes_features_W2 = W2(nodes_features)

# # Effectuez le calcul de right
# res = eta * nodes_features_W2.unsqueeze(2)
# right = torch.sum(res, dim=1)

# print(right.shape)

# batch_size = 1
# N_nodes = 3
# H = 2

# # Créez des tensors de forme (batch_size, N_nodes, N_nodes, H) et (batch_size, N_nodes, H) pour eta et nodes_features
# eta = torch.randn(batch_size, N_nodes, H)
# print(eta[0,1,1])
# print(eta.unsqueeze(2)[0,1,0,1])
# print(eta.unsqueeze(2).shape)

# print(eta)
# print(eta.unsqueeze(2))

# With Learnable Parameters
m = nn.BatchNorm1d(100)
# Without Learnable Parameters
m = nn.BatchNorm1d(100, affine=False)
input = torch.randn(20, 100)
output = m(input)
print(input)