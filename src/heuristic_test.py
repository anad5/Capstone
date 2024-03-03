# FOR THE COMBINED EDGES FILES AND IMPORTS ALL CIRCLE FILES 

import numpy as np
import matplotlib.pyplot as plt; plt.close('all')
import networkx as nx
from matplotlib.animation import FuncAnimation
import random as rd
import matplotlib.cm as cm
import matplotlib.colors as mcolors

from util_lae import generate_graph, animate_nodes, update_super, init_urns, pull_ball
from graph_dataALL import combine_graphs
from heuristic_functions import calc_centrality, calc_degree, calc_susceptibility

# folder with all edge files 
circle_files_path = 'circles'

# combined graph file with edges
combined_file_path = 'circles/facebook_combined.txt'

# combine all graphs into one
graph = combine_graphs(circle_files_path, combined_file_path)

# count the number of nodes in the graph to double check all circles are being brought in 
number_of_nodes = graph.number_of_nodes()
print("Number of nodes in the combined graph:", number_of_nodes)

#graph = nx.complete_graph(total_nodes)
#graph = generate_graph("./src/graph_data/Fig5_1_c_Adjacency_Matrix.txt")
time_steps = 10
delta_red = 1
delta_blue = 1
num_iters = 10
num_nodes = graph.number_of_nodes()

init_urns(graph)

#i initializing stuff for heuristic calculations 
#susceptibility_values = np.zeros((num_nodes, time_steps))
#degree_values = np.zeros((num_nodes, time_steps))
#centrality_values = np.zeros((num_nodes, time_steps))

# node IDs to indices 
node_to_index = {node: i for i, node in enumerate(graph.nodes())}

# susceptability is calculated every time step, degree and centrality arent? 
susceptibility_score = {}

for i in range(time_steps):
    update_super(graph)
    pull_ball(graph, delta_blue, delta_red, num_nodes)
    all_score = {}
    for node in graph.nodes():
        node_idx = node_to_index[node]
        score = calc_susceptibility(graph, node, 'red', 'total')
        all_score[node_idx] = score
    susceptibility_score[i] = all_score 

deg_score = {}
for node in graph.nodes():
    node_idx = node_to_index[node]
    score = calc_degree(graph, node)
    deg_score[node_idx] = score 

central_score = {} 
for node in graph.nodes():
    node_idx = node_to_index[node]
    score = calc_centrality(graph, node)
    central_score[node_idx] = score

# health = np.empty((num_nodes, time_steps+1))
# #for node in range(num_nodes):
# for node in graph.nodes():
#     health[node] = graph.nodes[node]['health']

# health = np.array(health)

# node_colors_r = health[:,:-1].T
# node_colors_r_1 = health[:,:-1]

# node IDs to indices 
node_to_index = {node: i for i, node in enumerate(graph.nodes())}

# Initialize the health array based on the number of nodes
num_nodes = len(graph.nodes())
health = np.empty((num_nodes, time_steps+1))

# Use the mapping when accessing the health array
for node in graph.nodes():
    index = node_to_index[node]  # Convert node ID to index
    health[index, :] = graph.nodes[node]['health']

# No need to convert to np.array, health is already a NumPy array
# health = np.array(health)

node_colors_r = health[:, :-1].T  # Use slicing to exclude the last column for all rows
node_colors_r_1 = health[:, :-1]  # This is the same as node_colors_r without transposition

# Assuming node_colors_template is to be used for some other purpose and unrelated to the node-to-index mapping
node_colors_template = np.random.randint(0, 100, size=(time_steps, num_nodes))


node_colors_template = np.random.randint(0, 100, size=(time_steps, num_nodes))


node_colors_test = np.ones((time_steps, num_nodes))

#normalize = mcolors.Normalize(vmin=health.min(), vmax=health.max())
normalize = mcolors.Normalize(vmin=0, vmax=1)
#colormap = cm.jet
colormap = mcolors.LinearSegmentedColormap.from_list("MyCmapName",["b","r"])
colormap = plt.colormaps.get_cmap('seismic')

scalarmappaple = cm.ScalarMappable(norm=normalize, cmap=colormap)
scalarmappaple.set_array(health[0,:])



animation = animate_nodes(graph, node_colors_r, scalarmappaple, colormap)
animation.save('gifs/heuristic_all.gif', writer='imagemagick', savefig_kwargs={'facecolor':'white'}, fps=1)