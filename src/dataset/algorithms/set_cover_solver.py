import copy
import numpy as np
import networkx as nx
import torch 
from scipy.optimize import Bounds, LinearConstraint, milp
from torch_geometric.utils.convert import to_networkx 


def solve_minimum_set_cover(num_vertices, num_edges, weight, edges):
    # Create the variables for MILP: one for each vertex
    c = np.array([weight[i] for i in range(num_vertices)])  # Objective coefficients (weights)

    # Prepare constraint matrices for each edge to cover
    A = np.zeros((num_edges, num_vertices))
    b = np.ones(num_edges)
    
    for idx in range(num_edges):
        for v in edges[idx]:
            A[idx, v] = 1

    # Constraints
    constraints = LinearConstraint(A, lb=b, ub=np.inf*np.ones(num_edges))     

    # Bounds: x_i are binary
    bounds = Bounds(0, 1)

    # Set options for integrality of variables: 1 means integer (binary in this case)
    integrality = np.ones(num_vertices, dtype=int)

    # Get the result
    result = milp(c=c, integrality=integrality, constraints=constraints, bounds=bounds)

    # Decode the solution
    if result.status == 0:
        optimal_weight = result.fun
        optimal_solution = result.x
    else:
        raise ValueError("No solution found: " + result.message)

    return optimal_solution, optimal_weight


def compute_from_nxgraph(graph):
    B = copy.deepcopy(graph)
    weight = B.graph['weights']
    num_vertices = len(weight)
    num_edges = B.number_of_nodes() - num_vertices
    edges = []
    for i in range(num_vertices, num_vertices + num_edges):
        edges.append(list(B.neighbors(i)))
    optimal_solutions, optimal_weight = solve_minimum_set_cover(num_vertices, num_edges, weight, edges)
    return optimal_solutions, optimal_weight


