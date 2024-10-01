import copy
import numpy as np
import networkx as nx
import torch 
from scipy.optimize import Bounds, LinearConstraint, milp, linprog 
from torch_geometric.utils.convert import to_networkx 


def solve_hitting_set_primal_int(T, c):
    # Get sizes 
    num_elements = len(c)
    num_sets = len(T)

    # Coefficients for the objective function and constraints 
    c = np.array(c) 
    A = np.zeros((num_sets, num_elements))
    for i in range(num_sets):
        for j in range(num_elements):
            if j in T[i]:
                A[i][j] = 1 

    # RHS of inequality constraints 
    lb = np.ones(num_sets)
    ub = np.full(num_sets, np.inf)
    
    # Bounds for the variables 
    x_bounds = Bounds(0, 1)

    # Constraint
    linear_constraint = LinearConstraint(A, lb, ub)

    # Solve the linear programming problem
    result = milp(c, integrality=np.ones(num_elements), bounds=x_bounds, constraints=[linear_constraint])

    # Get solutions 
    if result.status == 0:
        optimal_weight = result.fun
        optimal_solution = result.x
    else:
        raise ValueError("No solution found: " + result.message)

    return optimal_solution, optimal_weight


def solve_hitting_set_dual(T, c):
    # Get sizes 
    num_elements = len(c)
    num_sets = len(T)

    # Coefficients for the objective function and constraints 
    A = np.zeros((num_elements, num_sets))
    for i in range(num_elements):
        for j in range(num_sets):
            if i in T[j]:
                A[i][j] = 1 

    # RHS of inequality constraints 
    b = np.array(c)
    
    # Bounds for the variables 
    y_bounds = [(0, None) for _ in range(num_sets)]

    # Solve the linear programming problem 
    result = linprog(-np.ones(num_sets) , A_ub=A, b_ub=b, bounds=y_bounds, method='highs')

    # Get solutions 
    if result.success:
        optimal_weight = -result.fun
        optimal_solution = result.x
    else:
        raise ValueError("No solution found: " + result.message)

    return optimal_solution, optimal_weight


def compute_from_nxgraph(graph):
    B = copy.deepcopy(graph)
    num_nodes = B.number_of_nodes()
    c = B.graph['weights']
    num_elements = len(c)
    num_sets = num_nodes - num_elements
    T = [[] for _ in range(num_sets)]
    for i in range(num_sets):
        T[i] = list(B.neighbors(i + num_elements))
    primal_optimal_solution, primal_optimal_weight = solve_hitting_set_primal_int(T, c)
    dual_optimal_solution, dual_optimal_weight = solve_hitting_set_dual(T, c)
    return primal_optimal_solution, primal_optimal_weight, dual_optimal_solution, dual_optimal_weight


if __name__ == "__main__":
    from graph import generate_bipartite_graphs
    B = generate_bipartite_graphs(1, 10)[0]
    primal_optimal_solution, primal_optimal_weight, dual_optimal_solution, dual_optimal_weight = compute_from_nxgraph(B)
    breakpoint()
