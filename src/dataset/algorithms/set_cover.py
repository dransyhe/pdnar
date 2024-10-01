import copy
import torch
import numpy as np 
from torch_geometric.data import Data
from networkx.algorithms.bipartite.matrix import biadjacency_matrix

from src.dataset.data import BipartiteData
from src.dataset.algorithms.set_cover_solver import compute_from_nxgraph


def set_cover(graph):  
    B = copy.deepcopy(graph) 

    # Get optimal vertex covers 
    primal_optimal_solution, primal_optimal_weight = compute_from_nxgraph(B)
    primal_optimal_solution = torch.round(torch.tensor(primal_optimal_solution)).long()
    primal_optimal_weight = torch.tensor([primal_optimal_weight])

    # Graph info
    num_bipartite_nodes = B.number_of_nodes()
    w = B.graph['weights']
    num_nodes = len(w)
    num_edges = num_bipartite_nodes - num_nodes

    # Edge index
    row_order = [i + num_nodes for i in range(num_edges)]
    column_order = [i for i in range(num_nodes)]
    coo_matrix = biadjacency_matrix(B, row_order=row_order, column_order=column_order).tocoo()
    edge_index = torch.from_numpy(np.array([coo_matrix.row, coo_matrix.col])).long()

    # Initialization for the algorithm
    weights = B.graph['weights']
    w_p = weights.copy()
    node_mask = [False for _ in range(num_nodes)]
    edge_mask = [True for _ in range(num_edges)]

    # Hints and targets
    all_delta = [[0.0 for _ in range(num_edges)]]
    all_w_p = [weights.copy()]
    all_node_mask = [node_mask.copy()]
    all_edge_mask = [edge_mask.copy()]

    # Main algorithm: While edges remain
    while any(edge_mask):
        del_nodes = []
        delta = [0.0 for _ in range(num_edges)]

        # Calculate degree for each node
        d_p = [0. for _ in range(num_nodes)]  
        for n in range(num_nodes):
            if node_mask[n]:
                continue
            d_p[n] = B.degree(n)

        # For each remaining edge
        for e in range(num_edges):
            if not edge_mask[e]:
                continue
            min_delta = float('inf')
            for v in B.neighbors(num_nodes + e): 
                min_delta = min(min_delta, w_p[v] / d_p[v])
            if min_delta == float('inf'):
                min_delta = 0.
            delta[e] = min_delta 

        # For each remaining node
        for n in range(num_nodes):
            if node_mask[n]:
                continue
            sum = 0.
            for e in B.neighbors(n):
                if edge_mask[e - num_nodes]:
                    sum += delta[e - num_nodes]
            w_p[n] -= sum
            if w_p[n] <= weights[n] * 0.01:
                del_nodes.append(n)

        # Remove nodes and their incident edges
        for n in del_nodes:
            neighbors = list(B.neighbors(n)).copy()
            for e in neighbors:
                edge_mask[e - num_nodes] = False
                B.remove_node(e)
            B.remove_node(n)
            node_mask[n] = True

        # Record hints for each step
        all_delta.append(delta.copy())
        all_w_p.append(w_p.copy())
        all_node_mask.append(node_mask.copy())
        all_edge_mask.append(edge_mask.copy())

    # Use a time_mask to fill up for a maximal timesteps
    timesteps = len(all_delta)
    to_add = num_bipartite_nodes - timesteps
    assert to_add >= 0
    for _ in range(to_add):
        all_delta.append(all_delta[-1])
        all_w_p.append(all_w_p[-1])
        all_node_mask.append(all_node_mask[-1])
        all_edge_mask.append(all_edge_mask[-1])

    # Fill up optimal vertex covers to num_solutions 
    # num_solutions = len(optimal_vertex_covers)
    # optimal_mask = [torch.ones(num_nodes, dtype=torch.bool) for _ in range(num_solutions)]
    # for _ in range(10 - num_solutions):
    #     optimal_vertex_covers.append(optimal_vertex_covers[-1].clone())
    #     optimal_mask.append(torch.zeros(num_nodes, dtype=torch.bool))

    # Dimension transformation
    # Here, timestep = num_nodes * num_nodes is set to the maximum value
    delta = torch.tensor(all_delta).transpose(0, 1).unsqueeze(-1)  # delta = (num_edges, timesteps, 1)
    w_p = torch.tensor(all_w_p).transpose(0, 1).unsqueeze(-1)  # w_p = (num_nodes, timesteps, 1)
    node_mask = torch.tensor(all_node_mask).transpose(0, 1).unsqueeze(-1)  # node_mask = (num_nodes, timesteps)
    edge_mask = torch.tensor(all_edge_mask).transpose(0, 1).unsqueeze(-1)  # edge_mask = (num_edges, timesteps)

    return BipartiteData(y=delta,
                         x=w_p,
                         x_mask=node_mask.float(),
                         y_mask=edge_mask,
                         primal_optimal_solution=primal_optimal_solution,
                         primal_optimal_weight=primal_optimal_weight,
                         edge_index=edge_index)


if __name__ == '__main__':
    from src.dataset.graph import generate_bipartite_graphs

    B = generate_bipartite_graphs(1, 10)[0]
    data = set_cover(B)
    breakpoint()