import copy
import numpy as np
import networkx as nx
import torch 
from scipy.optimize import Bounds, LinearConstraint, milp
from torch_geometric.utils.convert import to_networkx 


def solve_minimum_vertex_cover(num_vertices, num_edges, weight, edges):
    # Create the variables for MILP: one for each vertex
    c = np.array([weight[i] for i in range(num_vertices)])  # Objective coefficients (weights)

    # Prepare constraint matrices for each edge (u, v): x_u + x_v >= 1
    A = np.zeros((num_edges, num_vertices))
    b = np.ones(num_edges)
    
    for idx, (u, v) in enumerate(edges):
        A[idx, u] = 1
        A[idx, v] = 1

    # Edge constraint: for each edge (u,v), x_u + x_v should be at least 1
    edge_constraint = LinearConstraint(A, lb=b, ub=np.inf*np.ones(num_edges))
    constraints = [edge_constraint]       

    # Bounds: x_i are binary
    bounds = Bounds(0, 1)

    # Set options for integrality of variables: 1 means integer (binary in this case)
    integrality = np.ones(num_vertices, dtype=int)

    # While loop until no optimal solution is found
    optimal_solutions = [] 
    optimal_weight = -1
    while True: 
        # Solve the MILP
        result = milp(c, bounds=bounds, constraints=constraints, integrality=integrality)

        # Decode the solution
        if result.status == 0: 
            solution = result.x
            vertex_cover = [0 for _ in range(num_vertices)]
            for i, p in enumerate(solution):
                if p > 0.5:
                    vertex_cover[i] = 1
            
            # Unique constraint: sum of x_i in the vertex cover < optimal_size
            size = vertex_cover.count(1)
            unique_constraint = LinearConstraint(vertex_cover, lb=0, ub=size - 1)
            constraints.append(unique_constraint)

            vertex_cover = torch.tensor(vertex_cover).to(torch.long)
            optimal_solutions.append(vertex_cover.clone())

            if optimal_weight == -1:
                # sum of weight[vertex_cover==1]
                optimal_weight = sum([weight[i] for i in range(num_vertices) if vertex_cover[i] == 1])
                # Optimal constraint: sum of weight(x_i==1) == optimal_weight
                optimal_constraint = LinearConstraint(weight, lb=optimal_weight, ub=optimal_weight)
                constraints.append(optimal_constraint)
            assert optimal_weight == sum([weight[i] for i in range(num_vertices) if vertex_cover[i] == 1])
        else:
            break 
        if len(optimal_solutions) == 10: 
            break

    return optimal_solutions, optimal_weight


def compute_from_nxgraph(graph):
    B = copy.deepcopy(graph)
    weight = B.graph['weights']
    num_vertices = len(weight)
    num_edges = B.number_of_nodes() - num_vertices
    edges = []
    for i in range(num_vertices, num_vertices + num_edges):
        u, v = list(B.neighbors(i))
        edges.append((u, v))
    edges = torch.tensor(edges).to(torch.long)
    optimal_solutions, optimal_weight = solve_minimum_vertex_cover(num_vertices, num_edges, weight, edges)
    return optimal_solutions, optimal_weight


if __name__ == "__main__":
    num_vertices = 5
    edges = [(0, 1), (0, 2), (1, 2), (1, 3), (1, 4), (2, 3), (3, 4)]  
    num_edges = len(edges)
    weights = [2, 3, 1, 1, 1] 
    optimal_solutions = solve_minimum_vertex_cover(num_vertices, num_edges, weights, edges)
    print(optimal_solutions)