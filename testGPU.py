import os
import json
import argparse
import time

import numpy as np

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn

import matplotlib
import matplotlib.pyplot as plt

import networkx as nx
from sklearn.utils.class_weight import compute_class_weight

from tensorboardX import SummaryWriter
from fastprogress import master_bar, progress_bar

# Remove warning
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from scipy.sparse import SparseEfficiencyWarning
warnings.simplefilter('ignore', SparseEfficiencyWarning)

from config import *
from utils.graph_utils import *
from utils.google_tsp_reader import GoogleTSPReader
from utils.plot_utils import *
# from models.gcn_model import ResidualGatedGCNModel
from utils.model_utils import *

import random
from models.model import MainModel

from train_functions import main, load_model

from utils.UCB import UCBTreeSearch, plot_predictions_UCBsearch, UCBSearch_with_batch, compute_optimal_tour_lenght, plot_predictions_two_methods, plot_predictions_random_comparison
from utils.UCB2 import UCBSearch_with_batch_UCB2
from utils.bs import beamSearch_with_batch
from utils.bs02 import beamSearch_with_batch02


config_path = "logs/deafult/config.json"
config = get_config(config_path)
print("Loaded {}:\n{}".format(config_path, config))

if torch.cuda.is_available():
    print("CUDA available")
    dtypeFloat = torch.cuda.FloatTensor
    dtypeLong = torch.cuda.LongTensor
    torch.cuda.manual_seed(1)
else:
    print("CUDA not available")
    dtypeFloat = torch.FloatTensor
    dtypeLong = torch.LongTensor
    torch.manual_seed(1)

load_path = "logs/deafult/checkpoint_epoch14.tar"
net = load_model(load_path, config)
print("model loaded")

max_trails_for_decoding = 1_000

net.eval()

batch_size = 1
num_nodes = config.num_nodes
num_neighbors = config.num_neighbors
beam_size = config.beam_size
test_filepath = config.test_filepath
dataset = iter(GoogleTSPReader(num_nodes, num_neighbors, batch_size, test_filepath))
batch = next(dataset)

with torch.no_grad():
    # Convert batch to torch Variables

    edges_k_means = Variable(torch.LongTensor(batch.edges).type(dtypeFloat), requires_grad=False)
    edges_dist = Variable(torch.FloatTensor(batch.edges_values).type(dtypeFloat), requires_grad=False)
    x_nodes = Variable(torch.LongTensor(batch.nodes).type(dtypeFloat), requires_grad=False)
    nodes_coord = Variable(torch.FloatTensor(batch.nodes_coord).type(dtypeFloat), requires_grad=False)
    y_edges = Variable(torch.LongTensor(batch.edges_target).type(dtypeFloat), requires_grad=False)
    y_nodes = Variable(torch.LongTensor(batch.nodes_target).type(dtypeFloat), requires_grad=False)
    
    # Compute class weights
    edge_labels = y_edges.cpu().numpy().flatten()
    edge_cw = compute_class_weight("balanced", classes=np.unique(edge_labels), y=edge_labels)
    print("Class weights: {}".format(edge_cw))
        
    # Compute class weights (if uncomputed)
    if type(edge_cw) != torch.Tensor:
        edge_labels = y_edges.cpu().numpy().flatten()
        edge_cw = compute_class_weight("balanced", classes=np.unique(edge_labels), y=edge_labels)
    
    # Forward pass
    y_preds, loss = net.forward(nodes_coord, edges_dist, edges_k_means, y_edges, edge_cw)
    
    y_UCB = 1- F.softmax(y_preds, dim=-1)
    
    y_UCB = y_UCB[:, :, :, 0]
    # TSP_UCB = UCBSearch_with_batch(y_UCB, edges_dist)
    bs = beamSearch_with_batch02(y_UCB, edges_dist,  max_trials=max_trails_for_decoding)
    TSP_UCB2 = UCBSearch_with_batch_UCB2(y_UCB, edges_dist, max_trials=max_trails_for_decoding)
    
                
    print("The gap to optimality for UCB is ", 100*compute_optimal_tour_lenght(edges_dist, y_edges, y_preds, TSP_UCB2), "‰")
    print("The gap to optimality for Beam Search is ", 100*compute_optimal_tour_lenght(edges_dist, y_edges, y_preds, bs), "‰")
    plot_predictions_two_methods(nodes_coord, edges_k_means, edges_dist, y_edges, y_preds, TSP_UCB2, bs, num_plots=batch_size)

