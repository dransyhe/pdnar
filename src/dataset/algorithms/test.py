
def check_data(data, algorithm):
    if "vertex_cover" in algorithm:
        nodes = [i for i in range(data.x.shape[0])]
        edges = [[] for _ in range(data.y.shape[0])]
        for i in range(data.edge_index.shape[1]):
            u, v = data.edge_index[:, i].tolist()
            edges[u].append(v)
        solution_opt = data.primal_optimal_solution.squeeze().tolist()
        solution_alg = data.x_mask[:, -1].squeeze().tolist()
        return check_vertex_cover(nodes, edges, solution_opt) and check_vertex_cover(nodes, edges, solution_alg)
    elif "set_cover" in algorithm:
        sets = [set() for _ in range(data.x.shape[0])]
        for i in range(data.edge_index.shape[1]):
            u, v = data.edge_index[:, i].tolist()
            sets[v].add(u)
        universe = set([i for i in range(data.y.shape[0])])
        solution_opt = data.primal_optimal_solution.squeeze().tolist()
        solution_alg = data.x_mask[:, -1].squeeze().tolist()
        return check_set_cover(sets, universe, solution_opt) and check_set_cover(sets, universe, solution_alg)
    elif algorithm == 'hitting_set':
        elements = [i for i in range(data.x.shape[0])]
        sets = [[] for _ in range(data.y.shape[0])]
        for i in range(data.edge_index.shape[1]):
            u, v = data.edge_index[:, i].tolist()
            sets[u].append(v)
        solution_opt = data.primal_optimal_solution.squeeze().tolist()
        solution_alg = data.x_mask[:, -1].squeeze().tolist()
        return check_hitting_set(elements, sets, solution_opt) and check_hitting_set(elements, sets, solution_alg)
    else:
        raise ValueError(f"Algorithm {algorithm} not supported.")


def check_vertex_cover(nodes, edges, solution):
    for edge in edges:
        if (not solution[edge[0]]) and (not solution[edge[1]]):
            return False
    return True 

def check_set_cover(sets, universe, solution):
    covered = set()
    for i, s in enumerate(sets):
        if solution[i]:
            covered |= s
    return covered == universe

def check_hitting_set(elements, sets, solution):
    for set in sets:
        found = False 
        for e in set:
            if solution[e]:
                found = True 
                break
        if not found:
            return False 
    return True 

    