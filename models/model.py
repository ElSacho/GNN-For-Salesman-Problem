import math

import torch
import torch.nn.functional as F
import torch.nn as nn

import numpy as np

from utils.model_utils import loss_edges


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Implement equation (2)
class NodesEncodeur(nn.Module):
    def __init__(self, hidden_dim):
        """_summary_

        Args:
            hidden_dim (int): the hidden_dim for the hidden layers
        """
        super(NodesEncodeur, self).__init__()
        self.A1 = nn.Linear(2, hidden_dim, True) # Each node has 2 positional values (x, y)
        
    def forward(self, nodes):
        """_summary_

        Args:
            nodes (tensor): tensor with the nodes of shape (batch_size, number_of_nodes, 2)

        Returns:
            tensor: (batch_size, number_of_nodes, H) embedding of the nodes (equation 2 of the paper) 
        """
                    
        # We transform the nodes to speed up the process
        batch_size, num_nodes, _ = nodes.size()
        nodes = nodes.view(batch_size * num_nodes, 2).contiguous()
        
        # We apply the equation (2) of the paper
        nodes = self.A1(nodes)
        
        # We reshape the data to the original shape
        nodes = nodes.view(batch_size, num_nodes, -1).contiguous()  # -1 to automaticaly adapt to hidden_dim
        return nodes
    
# Implement equation (3)
class EdgesEncodeur(nn.Module):
    def __init__(self, hidden_dim):
        """_summary_

        Args:
            hidden_dim (int): the hidden_dim for the hidden layers
        """
        super(EdgesEncodeur, self).__init__()
        # We check if we can divide by two the hidden_dim
        try:
            assert hidden_dim % 2 == 0
        except AssertionError:
            raise AssertionError("the hidden_dim cannot be divided by 2")
        self.A2 = nn.Linear(1, hidden_dim // 2, True)
        self.A3 = nn.Linear(1, hidden_dim // 2, True)
        

    def forward(self, edges, k_nearest_values):
        """_summary_

        Args:
            edges (tensor): tensor of the edges values (the distance between all the edges); shape : (batch_size, number_of_nodes, number_of_nodes)
            k_nearest_values (tensor): the tensor discribed in (3) in the paper which encode the k-nearest neighboors values; shape : (batch_size, number_of_nodes, number_of_nodes)

        Returns:
            tensor: shape (batch_size, N_nodes, N_nodes, H)
        """
        
        batch_size, number_of_nodes, number_of_nodes = edges.size()
        edges = edges.view(batch_size * number_of_nodes * number_of_nodes, 1).contiguous() # (B* N_nodes * N_nodes, 1)
        edges = self.A2(edges) # (B* N_nodes * N_nodes, H/2)
        edges = edges.view(batch_size, number_of_nodes, number_of_nodes, -1).contiguous() # (B, N_nodes, N_nodes, H/2)
        
        batch_size, number_of_nodes, number_of_nodes = k_nearest_values.size()
        k_nearest_values = k_nearest_values.view(batch_size * number_of_nodes * number_of_nodes, 1).contiguous() # (B* N_nodes * N_nodes, 1)
        k_nearest_values = self.A3(k_nearest_values) # (B* N_nodes * N_nodes, H/2)
        k_nearest_values = k_nearest_values.view(batch_size, number_of_nodes, number_of_nodes, -1).contiguous() # (B, N_nodes, N_nodes, H/2)
        
        edges_encoding = torch.concat((edges, k_nearest_values), dim = - 1) # (B, N_nodes, N_nodes, H)
        
        return edges_encoding
    
# Implement batch norm in equation (4)
class BatchNormNodes(nn.Module):
    def __init__(self, hidden_dim):
        """_summary_

        Args:
            hidden_dim (int) 
        """
        super(BatchNormNodes, self).__init__()
        self.W1 = nn.Linear(hidden_dim, hidden_dim, False) # In the paper there is no biais, but there is one on the repo
        self.W2 = nn.Linear(hidden_dim, hidden_dim, False) # In the paper there is no biais, but there is one on the repo
        self.batch_norm = nn.BatchNorm1d(hidden_dim, track_running_stats=False) # TODO : Need to check the track_running_stats=False
        
    def forward(self, nodes_features, edges_features, eps=1e-20):
        """_summary_

        Args:
            nodes_features (tensor): (batch_size, number_of_nodes, hidden_dim)
            edges_features (_type_): (batch_size, N_nodes, N_nodes, hidden_dim)
            eps (float) : a very small float that we add in the equation to calculate the dense attention map
            
        Returns:
            tensor: shape (batch_size, N_nodes, H)
        """
        # The left part of the equation in the BN
        x_left_equation = self.W1(nodes_features) # (batch_size, number_of_nodes, hidden_dim)
        
        # The right part of the equation in the BN
        # We start by calculating the eta (dense attention map)
        sigmoid_edges = torch.sigmoid(edges_features)  # (batch_size, number_of_nodes, number_of_nodes, hidden_dim) apply the sigmoid function to all the edges_features
        sum_sigmoid_edges = sigmoid_edges.sum(dim=2, keepdim=True)  # (batch_size, number_of_nodes, 1, hidden_dim)  we sum the sigmoid edges along all the j value (idx = 2)
        eta = sigmoid_edges / (sum_sigmoid_edges + eps) # (batch_size, number_of_nodes, number_of_nodes, hidden_dim) We get the dense attention map
        
        # Apply W2 to all the nodes 
        nodes_features_W2 = self.W2(nodes_features) # (batch_size, number_of_nodes, hidden_dim) 

        # We get the tensor inside the sum in equation (4)
        res = eta * nodes_features_W2.unsqueeze(1) # Hadamard product between (batch_size, number_of_nodes, number_of_nodes, hidden_dim) and (batch_size, 1, number_of_nodes, hidden_dim) -> (batch_size, number_of_nodes, number_of_nodes, hidden_dim)
        # res[B, i, j, H] = eta[B, i, j, H] * nodes_features_W2[B, 0, j, H] -> for the feature of the node j (the neighboors of i), we multiply those features by the feature map eta[i,j]
        sum_right_equation = torch.sum(res, dim=2) # (batch_size, number_of_nodes, hidden_dim) We make the sum of the result along the 'j' axis, represented here by dim=2
        
        # We make the sum of those values and take the batch norm :
        equ = x_left_equation + sum_right_equation # (batch_size, number_of_nodes, hidden_dim)
        
        equ_trans = equ.transpose(1, 2).contiguous()  # (batch_size, hidden_dim, number_of_nodes)
        equ_trans_bn = self.batch_norm(equ_trans) # (batch_size, hidden_dim, number_of_nodes)
        equ_trans_bn = equ_trans_bn.transpose(1, 2).contiguous()  # (batch_size, number_of_nodes, hidden_dim)
        return equ_trans_bn
        
# Implement batch norm in equation (5)
class BatchNormEdges(nn.Module):
    def __init__(self, hidden_dim):
        """_summary_

        Args:
            hidden_dim (int): hidden_dim
        """
        super(BatchNormEdges, self).__init__()
        self.W3 = nn.Linear(hidden_dim, hidden_dim, bias=False) # In the paper there is no biais, but there is one on the repo
        self.W4 = nn.Linear(hidden_dim, hidden_dim, bias=False) # In the paper there is no biais, but there is one on the repo
        self.W5 = nn.Linear(hidden_dim, hidden_dim, bias=False) # In the paper there is no biais, but there is one on the repo
        self.batch_norm = nn.BatchNorm2d(hidden_dim, track_running_stats=False) # 2D because we are looking at 2 information (edges_ij)

    def forward(self, nodes_features, edges_features):
        """_summary_

        Args:
            nodes_features (tensor): (batch_size, number_of_nodes, hidden_dim)
            edges_features (tensor): (batch_size, number_of_nodes, number_of_nodes, hidden_dim)
        
        Returns:
            tensor: shape (batch_size, number_of_nodes, number_of_nodes, H)
        """
        
        edges_features_W3 = self.W3(edges_features) # (batch_size, num_nodes, num_nodes, hidden_dim)
        
        node_features_W4 = self.W4(nodes_features)  # (batch_size, num_nodes, num_nodes, hidden_dim)
        node_features_W5 = self.W5(nodes_features)  # (batch_size, num_nodes, num_nodes, hidden_dim)

        # Calculer le tenseur e_
        e_f = edges_features_W3 + node_features_W4.unsqueeze(2) + node_features_W5.unsqueeze(1) # (batch_size, num_nodes, num_nodes, hidden_dim)
        e_f_trans = e_f.transpose(1, 3).contiguous()  # (batch_size, hidden_dim, num_nodes, num_nodes)
        e_f_trans_bn = self.batch_norm(e_f_trans) # (batch_size, hidden_dim, num_nodes, num_nodes)
        e_f_bn = e_f_trans_bn.transpose(1, 3).contiguous()  # (batch_size, num_nodes, num_nodes, hidden_dim)
        
        tensors = [edges_features_W3, node_features_W4, node_features_W5, e_f, e_f_trans, e_f_trans_bn, e_f_bn]
        for tensor in tensors:
            if not tensor.is_contiguous():
                print("A")
                # Result : ok
        
        return e_f_bn
        
# Implement equation (4) & (5)
class HiddenLayer(nn.Module):
    def __init__(self, hidden_dim):
        """_summary_

        Args:
            hidden_dim (int)
        """
        super(HiddenLayer, self).__init__()
        self.hidden_dim = hidden_dim
        self.bnNodes = BatchNormNodes(self.hidden_dim)
        self.bnEdges = BatchNormEdges(self.hidden_dim)

    def forward(self, nodes_features, edges_features):
        """_summary_

        Args:
            nodes_features (tensor): (batch_size, number_of_nodes, hidden_dim) 
            edges_features (tensor): (batch_size, number_of_nodes, number_of_nodes, hidden_dim)

        Returns:
            tensor, tensor: shapes (batch_size, number_of_nodes, hidden_dim) and (batch_size, number_of_nodes, number_of_nodes, hidden_dim)
        """        
        bn_nodes = self.bnNodes(nodes_features, edges_features)
        bn_edges = self.bnEdges(nodes_features, edges_features)
        
        nodes_features = nodes_features + F.relu(bn_nodes)
        edges_features = edges_features + F.relu(bn_edges)
        
        tensors = [bn_nodes, bn_edges, nodes_features, edges_features]
        for tensor in tensors:
            if not tensor.is_contiguous():
                print("A")
                # result : OK
                
        return nodes_features, edges_features

# MLP to get the probabilities for each nodes to be selected - see equation (6)
class MLP(nn.Module):
    def __init__(self, hidden_dim, output_dim, L=2):
        super(MLP, self).__init__()
        self.L = L
        U = []
        for layer in range(self.L - 1):
            U.append(nn.Linear(hidden_dim, hidden_dim, True))
        self.U = nn.ModuleList(U)
        self.V = nn.Linear(hidden_dim, output_dim, True)
        self.output_dim = output_dim

    def forward(self, x):
        """
        Args:
            x: Input features (batch_size, hidden_dim)

        Returns:
            y: Output predictions (batch_size, output_dim)
        """
        Ux = x
        for U_i in self.U:
            Ux = U_i(Ux)  # B x H
            Ux = F.relu(Ux)  # B x H
        y = self.V(Ux)  # B x O
        
        # Y : (batch_size, n_nodes, n_nodes, n_labels=2)
        # Y[b,i,j,1] = P(for the graph b in the batch size, the node i is linked to the node j for the TSP)
        # Y[b,i,j,0] = P(for the graph b in the batch size, the node i is NOT linked to the node j for the TSP)
        
        return y
    
class MainModel(nn.Module):
    def __init__(self, config):
        super(MainModel, self).__init__()
        self.num_nodes = config.num_nodes
        self.node_dim = config.node_dim
        self.voc_nodes_in = config['voc_nodes_in']   # TODO
        self.voc_nodes_out = config['num_nodes']     # TODO
        self.voc_edges_in = config['voc_edges_in']   # TODO
        self.voc_edges_out = config['voc_edges_out'] # TODO
        self.hidden_dim = config['hidden_dim']
        self.num_layers = config['num_layers'] 
        self.mlp_layers = config['mlp_layers']       # TODO
        
        # Encoding layers
        self.NodesEncoding = NodesEncodeur(self.hidden_dim) 
        self.EdgesEncoding = EdgesEncodeur(self.hidden_dim)
        # All the hidden layers (repeat equations (4) and (5))
        gcn_layers = [HiddenLayer(self.hidden_dim) for _ in range(self.num_layers)]
        self.gcn_layers = nn.ModuleList(gcn_layers)
        
        self.mlp_edges = MLP(self.hidden_dim, self.voc_edges_out, self.mlp_layers)
        
    def forward(self, nodes_coord, edges_dist, edges_k_means, y_edges, edge_cw):
        """_summary_

        Args:
            nodes_coord (tensor): (batch_size, number_of_nodes, 2)
            edges_dist (tensor): edge distance matrix (batch_size, num_nodes, num_nodes)
            edges_k_means (tensor) : k_nearest value for edges tensor (batch_size, num_nodes, num_nodes)
            y_edges (tensor): Targets for edges (batch_size, num_nodes, num_nodes)
            edge_cw (tensor): TODO
            
        Returns:
            y_pred_edges: Output predictions (batch_size, output_dim)
        """
        # Encoding the values        
        nodes_features = self.NodesEncoding(nodes_coord) # (batch_size, number_of_nodes, H) embedding of the nodes (equation 2 of the paper)
        edges_features = self.EdgesEncoding(edges_dist, edges_k_means) # (batch_size, number_of_nodes, number_of_nodes, H) embedding of the nodes (equation 3 of the paper)
        tensors = [nodes_features, edges_features]
        for tensor in tensors:
            if not tensor.is_contiguous():
                print("A")


        # Passing through the hidden layers
        for layer in self.gcn_layers:
            nodes_features, edges_features = layer(nodes_features, edges_features)  # (batch_size, number_of_nodes, H)  and # (batch_size, number_of_nodes, number_of_nodes, H)
            tensors = [nodes_features, edges_features]
            for tensor in tensors:
                if not tensor.is_contiguous():
                    print("A")
                    # Result OK

        # Get the probability graph 
        y_pred_edges = self.mlp_edges(edges_features)
        y_edges = y_edges.long()
    
        edge_cw = torch.Tensor(edge_cw).type(torch.FloatTensor)  # Convert to tensors
        loss = loss_edges(y_pred_edges, y_edges, edge_cw)
        
        tensors = [y_pred_edges, loss]
        for tensor in tensors:
            if not tensor.is_contiguous():
                print("A")
                # RESULT : OK
        
        return y_pred_edges, loss
        

        
        