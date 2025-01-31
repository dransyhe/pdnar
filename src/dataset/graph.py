import math
import numpy as np
import networkx as nx
import random
from itertools import combinations



def generate_bipartite_graphs_from_barabasi(num_graphs, num_nodes):
    graphs = []
    m_choices = [i for i in range(1, 10)]
    for _ in range(num_graphs):
        m = random.sample(m_choices, k=1)[0]
        G = nx.barabasi_albert_graph(num_nodes, m)
        B = convert_to_bipartite(G)
        # Assign random weights to RHS nodes
        node_weight = []
        for _ in range(num_nodes):
            node_weight += [random.uniform(1e-10, 1)]
        B.graph['weights'] = node_weight
        graphs.append(B)
    return graphs


def generate_bipartite_graphs(num_graphs, num_nodes, b=5):
    graphs = []
    for _ in range(num_graphs):
        B = nx.Graph()
        n = max(3, random.randint(num_nodes // 4, num_nodes // 4 * 3))
        m = num_nodes - n 
        offline_nodes = range(n)
        B.add_nodes_from(offline_nodes, bipartite=1)  
        
        def degree_probability(node, B):
            return B.degree(node) + 1  # Add 1 to avoid division by zero 
        
        for online_node in range(n, n + m):
            B.add_node(online_node, bipartite=0)  
            
            offline_probabilities = [degree_probability(node, B) for node in offline_nodes]
            total_prob = sum(offline_probabilities)
            probabilities = [prob / total_prob for prob in offline_probabilities]
            selected_offline_nodes = random.choices(offline_nodes, probabilities, k=b)
            
            B.add_edges_from((online_node, offline_node) for offline_node in selected_offline_nodes)

        node_weight = []
        for _ in range(n):
            node_weight += [random.uniform(1e-10, 1)]
        B.graph['weights'] = node_weight

        graphs.append(B)

    return graphs


def generate_test_graphs(graph_type, num_graphs, num_nodes):
    if graph_type == 'erdos':
        return generate_bipartite_graphs_from_erdos(num_graphs, num_nodes)
    elif graph_type == 'star':
        return generate_star_graph(num_graphs, num_nodes)
    elif graph_type == 'lobster':
        return generate_lobster_graph(num_graphs, num_nodes)
    elif graph_type == '3-connected':
        return generate_bipartite_graph_from_3_connected_cubic_planar_graph(num_graphs, num_nodes)
    else:
        raise ValueError(f"Graph type {graph_type} not supported")


def generate_bipartite_graphs_from_erdos(num_graphs, num_nodes):
    graphs = []
    for _ in range(num_graphs):
        edge_prob = random.uniform(0.2, 0.8)
        G = nx.erdos_renyi_graph(num_nodes, edge_prob)
        B = convert_to_bipartite(G)
        # Assign random weights to RHS nodes
        node_weight = []
        for _ in range(num_nodes):
            node_weight += [random.uniform(1e-10, 1)]
        B.graph['weights'] = node_weight
        graphs.append(B)
    return graphs


def generate_star_graph(num_graphs, num_nodes):
    graphs = []
    for _ in range(num_graphs):
        # Partition the nodes randomly into 1 to 5 sets
        sets = [[] for _ in range(random.randint(1, 5))]
        nodes = list(range(num_nodes))
        random.shuffle(nodes)
        for node in nodes:
            random.choice(sets).append(node)
        node_sets = [s for s in sets if s]

        # Generate a star graph for each node set
        star_graphs = []
        for node_set in node_sets:
            if len(node_set) == 1:
                G = nx.Graph()
                G.add_node(node_set[0])
            else:
                G = nx.star_graph(node_set[0:len(node_set)])
            star_graphs += [G]
        
        # Combine the star graphs 
        G = nx.Graph()
        for star in star_graphs:
            G = nx.compose(G, star)

        # Add random edges between the star graphs to make connected graph
        set_indices = list(range(len(node_sets)))
        random.shuffle(set_indices)
        for i in range(len(set_indices) - 1):
            set1 = node_sets[set_indices[i]]
            set2 = node_sets[set_indices[i + 1]]
            node1 = random.choice(set1)
            node2 = random.choice(set2)
            G.add_edge(node1, node2)

        # Relabel nodes to be in order
        mapping = {node: i for i, node in enumerate(G.nodes)}
        G = nx.relabel_nodes(G, mapping)

        B = convert_to_bipartite(G)
        # Assign random weights to RHS nodes
        node_weight = []
        for _ in range(num_nodes):
            node_weight += [random.uniform(1e-10, 1)]
        B.graph['weights'] = node_weight
        graphs.append(B)
    return graphs


def generate_lobster_graph(num_graphs, num_nodes):
    graphs = []
    for _ in range(num_graphs):
        G = nx.empty_graph(num_nodes)
        backbone = np.random.randint(low=1, high=num_nodes)
        branches = np.random.randint(low=backbone + 1, high=num_nodes + 1)
        # Generate the backbone stalk
        for i in range(1, backbone):
            G.add_edge(i - 1, i)
        # Add branches to the backbone
        for i in range(backbone, branches):
            G.add_edge(i, np.random.randint(backbone))
        # Attach rest of nodes to the branches
        for i in range(branches, num_nodes):
            G.add_edge(i, np.random.randint(low=backbone, high=branches))
        B = convert_to_bipartite(G)
        node_weight = []
        for _ in range(num_nodes):
            node_weight += [random.uniform(1e-10, 1)]
        B.graph['weights'] = node_weight
        graphs += [B]
    return graphs


def generate_bipartite_graph_from_3_connected_cubic_planar_graph(num_graphs, num_nodes, min_girth=4, max_attempts=100):
    graphs = []
    for _ in range(num_graphs):
        for attempt in range(100):
            # Generate a random 3-regular graph
            G = nx.random_regular_graph(3, num_nodes)
            
            # Check if the graph is planar
            is_planar, _ = nx.check_planarity(G)
            
            # Check if the graph is 3-connected
            if is_planar and (nx.node_connectivity(G) >= 3):
                break
        B = convert_to_bipartite(G)
        # Assign random weights to RHS nodes
        node_weight = []
        for _ in range(num_nodes):
            node_weight += [random.uniform(1e-10, 1)]
        B.graph['weights'] = node_weight
        graphs.append(B)
    return graphs


def convert_to_bipartite(G):
    B = nx.Graph()
    n = G.number_of_nodes()
    m = G.number_of_edges()
    
    B.add_nodes_from(range(n), bipartite=1)
    B.add_nodes_from(range(n, n + m), bipartite=0)
    
    # Add edges between the left and right nodes
    edge_index = n
    for i, edge in enumerate(G.edges()):
        u, v = edge
        B.add_edge(edge_index, u)
        B.add_edge(edge_index, v)
        edge_index += 1
    
    return B


if __name__ == '__main__':
    graphs = generate_bipartite_graphs(1, 10)
    for G in graphs:
        print(G.nodes)
        print(G.edges)
        print(nx.get_node_attributes(G, 'weight'))
        print()
