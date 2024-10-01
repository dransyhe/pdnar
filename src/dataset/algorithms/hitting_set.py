import copy
import torch
import numpy as np
from torch_geometric.utils import to_torch_coo_tensor 
from networkx.algorithms.bipartite.matrix import biadjacency_matrix

from src.dataset.data import BipartiteData
from src.dataset.algorithms.hitting_set_solver import compute_from_nxgraph


def hitting_set(graph):
    B = copy.deepcopy(graph) 

    # Get optimal solutions 
    primal_optimal_solution, primal_optimal_weight, dual_optimal_solution, dual_optimal_weight = compute_from_nxgraph(B)
    primal_optimal_solution = torch.round(torch.tensor(primal_optimal_solution)).long()
    primal_optimal_weight = torch.tensor([primal_optimal_weight])
    dual_optimal_solution = torch.tensor(dual_optimal_solution)
    dual_optimal_weight = torch.tensor([dual_optimal_weight])

    # Get T, E, c
    num_nodes = B.number_of_nodes()
    ce = B.graph['weights']
    num_elements = len(ce)
    num_sets = num_nodes - num_elements
    # nodes 0 to (num_elements - 1) are RHS (E)
    # nodes num_elements to (num_nodes - 1) are LHS (T)
    T = [[] for _ in range(num_sets)]
    for i in range(num_sets):
        T[i] = list(B.neighbors(i + num_elements))

    # Get biadjacency matrix for the bipartite graph
    row_order = [i + num_elements for i in range(num_sets)]
    column_order = [i for i in range(num_elements)]
    coo_matrix = biadjacency_matrix(B, row_order=row_order, column_order=column_order).tocoo()
    edge_index = torch.from_numpy(np.array([coo_matrix.row, coo_matrix.col])).long()

    # Violation oracle 
    def _violation(A, T):
        V = [False for _ in range(len(T))]
        for i in range(len(T)):
            if len(A.intersection(T[i])) == 0:
                V[i] = True
        return V 

    # Initialize variables 
    y = [0. for _ in range(num_sets)]
    A = set([])
    x = [0 for _ in range(num_elements)]  # this is also x_mask
    y_mask = [True for _ in range(num_sets)]  
    remain_ce = ce.copy()

    # Initialize saved data 
    all_x = [x.copy()] 
    all_y = [y.copy()] 
    all_V = [[True for _ in range(num_sets)]]
    all_remain_ce = [remain_ce.copy()]
    all_y_mask = [y_mask.copy()]
    all_delta = [[0. for _ in range(num_sets)]]
    all_eps = [[0.]]
    all_eps_mask = [[True]]

    # Primal-dual algorithm for hitting set 
    added_seq = []
    while True:  
        # Compute the violation set; terminate when it's empty, i.e. A is feasible given T
        V = _violation(A, T) 
        if not any(V):
            break 

        # Increase y[k] uniformly for k in V 
        # Get degrees
        degree_e = [0 for _ in range(num_elements)]
        for e in range(num_elements):
            if e in B.nodes():
                degree_e[e] = B.degree[e]
        # Get the minimum from remain_ce[e] / degree_e[e]
        delta = [float('inf') for _ in range(num_sets)]
        for k in range(num_sets):
            if V[k]:
                for e in T[k]:
                    if not x[e]: 
                        delta[k] = min(delta[k], remain_ce[e] / degree_e[e])
        eps = min(delta) 
        assert eps != float('inf')
        for k in range(num_sets):
            if delta[k] == float('inf'):
                delta[k] = 0.

        # Get the index of element that meets the constraint (and to add into A)
        for e in range(num_elements): # TODO: tie breaking; currently finds the first min
            if degree_e[e] != 0:
                if abs(remain_ce[e] / degree_e[e] - eps) < 1e-10:
                    l = e 
                    assert x[l] == 0 
                    break 

        # Updates y[k] for all k in V
        for k in range(num_sets):
            if V[k]:
                y[k] += eps 

        # Add e_l to the set A
        A.add(l)
        added_seq.append(l)
        x[l] = 1.  

        # Update remain_ce
        for e in range(num_elements):
            if degree_e[e] > 0:
                remain_ce[e] -= eps * degree_e[e]

        # Remove e_l and its connected y_k from the bipartite graph 
        remove_nodes = [l]
        for k in list(B.neighbors(l)):
            y_mask[k - num_elements] = False
            remove_nodes += [k]
        B.remove_nodes_from(remove_nodes)

        # Save data for each timestep 
        all_x.append(x.copy())
        all_y.append(y.copy())
        all_V.append(V.copy())
        all_remain_ce.append(remain_ce.copy())
        all_y_mask.append(y_mask.copy())
        all_delta.append(delta.copy())
        all_eps.append([eps])
        all_eps_mask.append([True])

    assert B.number_of_edges() == 0
    assert all_y_mask[-1] == [False for _ in range(num_sets)]
    assert all([all([v > 0 - 1e-10 for v in remain_ce]) for remain_ce in all_remain_ce])
    assert all([all([v > 0 - 1e-10 < 1e-10 for v in delta]) for delta in all_delta])
    assert all([all([v > 0 - 1e-10 < 1e-10 for v in eps]) for eps in all_eps])

    # Use a time_mask to fill up for a maximum of #edges = num_sets * num_elements timesteps
    timesteps = len(all_x)
    to_add = num_nodes - timesteps
    assert to_add >= 0
    for _ in range(to_add):
        all_x.append(all_x[-1])
        all_y.append(all_y[-1])
        all_V.append(all_V[-1])
        all_remain_ce.append(all_remain_ce[-1])
        all_delta.append(all_delta[-1])
        all_eps.append(all_eps[-1])
        all_eps_mask.append([False])
        all_y_mask.append(all_y_mask[-1])

    # Dimension transformation
    # Here, timestep = num_nodes is set to the maximum value (maximally is num_elements, but it differs across graphs)
    x = torch.tensor(all_x).transpose(0, 1).unsqueeze(-1)  # x = (num_elements, timesteps, 1)
    y = torch.tensor(all_y).transpose(0, 1).unsqueeze(-1)  # y = (num_sets, timesteps, 1)
    V = torch.tensor(all_V).transpose(0, 1).unsqueeze(-1)  # V = (num_sets, timesteps, 1)
    remain_ce = torch.tensor(all_remain_ce).transpose(0, 1).unsqueeze(-1)  # remain_ce = (num_elements, timesteps, 1)
    delta = torch.tensor(all_delta).transpose(0, 1).unsqueeze(-1)  # delta = (num_sets, timesteps, 1)
    eps = torch.tensor(all_eps).transpose(0, 1).unsqueeze(-1)  # eps = (1, timesteps, 1)
    eps_mask = torch.tensor(all_eps_mask).transpose(0, 1).unsqueeze(-1)  # eps_mask = (1, timesteps, 1)
    y_mask = torch.tensor(all_y_mask).transpose(0, 1).unsqueeze(-1)  # y_mask = (num_sets, timesteps, 1)

    return BipartiteData(x_mask=x,
                         x=remain_ce,
                         y=delta,
                         eps=eps,
                         eps_mask=eps_mask,
                         y_mask=y_mask,
                         edge_index=edge_index,
                         primal_optimal_solution=primal_optimal_solution, 
                         primal_optimal_weight=primal_optimal_weight, 
                         dual_optimal_solution=dual_optimal_solution, 
                         dual_optimal_weight=dual_optimal_weight) 


if __name__ == '__main__':
    from graph import generate_bipartite_graphs
    import random 
    random.seed(42)

    for _ in range(1000):
        B = generate_bipartite_graphs(1, 10)[0]
        data = hitting_set(B)
    breakpoint()
