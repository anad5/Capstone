import networkx as nx

# Function to load edges from a file and create a graph
def load_graph(edges_file_path):
    G = nx.Graph()
    with open(edges_file_path, 'r') as file:
        for line in file:
            edge = line.strip().split(' ')
            G.add_edge(edge[0], edge[1])
    return G

# Example usage
edges_file_path = "/Users/madeline/Downloads/archive (1)/facebook/facebook/414.edges"
G = load_graph(edges_file_path)

import matplotlib.pyplot as plt
import networkx as nx

# Assuming G is your graph object
# G = load_graph(edges_file_path)  # This line should already be in your script

# Basic visualization
plt.figure(figsize=(10, 10))  # Set the size of the plot
nx.draw(G, with_labels=True, node_size=50, node_color="lightblue", edge_color="gray")
plt.title("Network Visualization")
plt.show()
