from pyvis.network import Network
import networkx as nx
from IPython.display import display, IFrame

# Function to load a graph from an edges file
def load_graph(edges_file_path):
    G = nx.Graph()
    with open(edges_file_path, 'r') as file:
        for line in file:
            edge = line.strip().split(' ')
            G.add_edge(edge[0], edge[1])
    return G

edges_file_path1 = "/Users/madeline/Downloads/archive (1)/facebook/facebook/414.edges"
G = load_graph(edges_file_path1)

# Initialize the Pyvis network with inline resources for Jupyter notebook compatibility
# If running as a standalone script, you might set notebook=False
net = Network(notebook=True, height='750px', width='100%', cdn_resources='in_line')
net.from_nx(G)
net.show('my_network.html')
display(IFrame('my_network.html', width='100%', height='750px'))



