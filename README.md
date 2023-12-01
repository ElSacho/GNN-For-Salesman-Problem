# Traveling Salesman Problem Solver

Project realised for the course "Geometrical Data Analysis" from Jean Feydy during the first semester of the MVA 2023-2024.
The goal was to produce a report on the paper "An Efficient Graph Convolutional Network Technique for the Travelling Salesman Problem", Chaitanya K. Joshi, Thomas Laurent, Xavier Bresson, https://arxiv.org/abs/1906.01227
In this report, we tried to go beyond the subject implementing new methods based on bandit theory to find the shortest tour of a TSP. 

## Overview

The Traveling Salesman Problem (TSP) is a classic optimization challenge with applications in computer science, operations research, and mathematics. This combinatorial problem involves finding the shortest possible tour that visits each city exactly once and returns to the starting city, given a set of cities and the distances between them. TSP is NP-hard, making the development of efficient algorithms for sub-optimal solutions crucial in various research areas, including DNA sequencing.

## Project Components

This repository explores a framework for obtaining sub-optimal solutions for the TSP. The approach consists of two main components:

1. **Graph Neural Network Training:** We train a graph neural network on TSP instances with fixed sizes. The network predicts the probability of each edge being included in the optimal TSP solution, effectively learning the underlying patterns in the data.

2. **Path Optimization:** Using the probabilities obtained from the neural network, we implemented sevaral decoding strategies to find the shortest path. 

## Methodology

Our contribution involves a re-implementation of the original methods, specifically the convolutional graph neural network and the beam search. Additionally, we introduce a novel decoding method inspired by the upper confidence bound for tree search. This decoding strategy outperforms traditional methods, reducing the gap to the optimal tour significantly.

## Results

Our experiments reveal promising outcomes. Despite limitations in CPU capacities affecting neural network training, our decoding strategy achieves a remarkable 11.08% gap to the optimal tour for a 50-node problem, compared to the 47.76% gap from beam search. Moreover, we successfully scale our model from a TSP20 to a 50-node problem, achieving a 10.09% gap to optimal, surpassing the original paper's 34.46% gap.

## Conclusion and Future Work

While the original paper's strategy proves efficient for small TSP instances, it lacks robustness for larger-scale problems (nodes > 100). The non-autoregressive model accelerates computation but becomes specific to a certain number of nodes. Our novel scaling approach shows promise, but further research, including GPU-based training, is needed for comprehensive evaluation and optimization.

Feel free to explore the code and experiments in this repository, and contribute to advancing the state of the art in solving the Traveling Salesman Problem.
