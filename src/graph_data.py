import networkx as nx

def read_edges(filename):
    G = nx.Graph()
    with open(filename, 'r') as f:
        for line in f:
            src, dst = map(int, line.strip().split())
            G.add_edge(src, dst)
    return G

def read_features(filename, graph):
    with open(filename, 'r') as f:
        for line in f:
            parts = line.strip().split()
            node_id = int(parts[0])  
            features = map(int, parts[1:])  # Rest are the features
            if node_id not in graph:
                graph.add_node(node_id)
            for j, feat in enumerate(features):
                graph.nodes[node_id]['feature_{}'.format(j)] = feat


def read_egofeat(filename, graph, ego_node):
    with open(filename, 'r') as f:
        features = map(int, f.readline().strip().split())
        # Ensure the ego_node exists in the graph
        if ego_node not in graph:
            graph.add_node(ego_node)
        for j, feat in enumerate(features):
            graph.nodes[ego_node]['feature_{}'.format(j)] = feat


def load_graph(edge_path, feature_path, egofeat_path, ego_node=0):
    graph = read_edges(edge_path)
    read_features(feature_path, graph)
    read_egofeat(egofeat_path, graph, ego_node)
    return graph
