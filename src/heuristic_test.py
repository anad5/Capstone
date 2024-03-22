# FOR THE COMBINED EDGES FILES AND IMPORTS ALL CIRCLE FILES 

# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt; plt.close('all')
import networkx as nx
from matplotlib.animation import FuncAnimation
import random as rd
import matplotlib.cm as cm
import matplotlib.colors as mcolors

# Import functions from custom modules
from util import generate_graph, animate_nodes, update_super, init_urns, pull_ball, plot_health
from graph_dataALL import combine_graphs
from heuristic_functions import calc_centrality, calc_degree, calc_susceptibility, score_midpoint

# Folder path containing all edge files
circle_files_path = 'circles'

# Combined graph file with edges
combined_file_path = 'circles/facebook_combined.txt'

# Combine all graphs into one
graph = combine_graphs(circle_files_path, combined_file_path)

# Count the number of nodes in the graph to double-check all circles are being brought in
number_of_nodes = graph.number_of_nodes()
print("Number of nodes in the combined graph:", number_of_nodes)

# Flag to indicate whether memory is enabled
memory_flag = True
graph.graph['memory_flag'] = memory_flag

# Define simulation parameters
time_steps = 100
delta_red = 1
delta_blue = 1
init_red = 10
init_blue = 10
num_nodes = graph.number_of_nodes()

# Generate random memory values for each node
memory_list = np.random.default_rng().normal(9, 3, size=num_nodes).astype(int)
memory_list = np.where(memory_list <= 0, 1, memory_list)

# Generate random initial red and blue balls for each node
init_red_list = np.random.default_rng().normal(10, 2, size=num_nodes).astype(int)
init_red_list = np.where(init_red_list <= 0, 1, init_red_list)

init_blue_list = np.random.default_rng().normal(10, 2, size=num_nodes).astype(int)
init_blue_list = np.where(init_blue_list <= 0, 1, init_blue_list)

# Initialize urns with initial values and memory
init_urns(graph, init_red, init_blue, memory_list=memory_list, init_blue_list=init_blue_list, init_red_list=init_red_list)

# Create a mapping from node IDs to indices
node_to_index = {node: i for i, node in enumerate(graph.nodes())}

# Calculate degree and centrality scores outside the time step loop
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

# Initialize a dictionary to store susceptibility scores at each time step
all_scores = {} 
beta = 1
gamma = 1
alpha = 1

# Run simulation for each time step
for i in range(time_steps):
    # Update super urns after pulling balls
    update_super(graph)
    
    # Pull balls from urns
    pull_ball(graph, delta_blue, delta_red)
    
    # Calculate susceptibility score for each node
    suscept_score = {}
    for node in graph.nodes():
        node_idx = node_to_index[node]
        score = calc_susceptibility(graph, node, 'red', 'total')
        suscept_score[node_idx] = score
        
    # Calculate heuristic scores for each node
    all_scores[i] = {}
    for node in graph.nodes():
        node_idx = node_to_index[node]
        degree_score = deg_score[node_idx] 
        centrality_score = central_score[node_idx]  
        susceptibility_score = suscept_score[node_idx]
            
        # Calculate the combined score using heuristic score equation (found in thesis, design solution section)
        combined = beta * degree_score + gamma * centrality_score - alpha * susceptibility_score
        all_scores[i][node_idx] = combined

    # Select nodes with the highest combined scores as the midpoints
    nodes_midpoint = score_midpoint(all_scores[i])

    # Initialize budgets for red and blue balls
    budget_red = 500
    budget_blue = 500

    # Distribute red balls to nodes with highest combined scores until budget is exhausted
    for node_idx in nodes_midpoint:
        if budget_red > 0:
            graph.nodes[node_idx]['red'] += delta_red
            graph.nodes[node_idx]['total'] += delta_red
            budget_red -= delta_red 
        else:
            break 

    # Distribute remaining budget for blue balls to all nodes
    for node in graph.nodes(): 
        graph.nodes[node_idx]['blue'] += delta_blue
        graph.nodes[node_idx]['total'] += delta_blue
        budget_blue -= delta_blue

# Plot health of the network over time
health = np.empty((num_nodes, time_steps+1))
for node in graph.nodes():
    index = node_to_index[node]
    health[index, :] = graph.nodes[node]['health']
health = np.array(health)
plot_health(graph, health)

# Convert node IDs to indices for animation
node_to_index = {node: i for i, node in enumerate(graph.nodes())}

# Extract health data for animation
node_colors_r = health[:, :-1].T  

# Set up colormap and scalar mappable for animation
normalize = mcolors.Normalize(vmin=0, vmax=1)
colormap = plt.colormaps.get_cmap('seismic')
scalarmappaple = cm.ScalarMappable(norm=normalize, cmap=colormap)
scalarmappaple.set_array(health[0,:])

# Create and save animation
animation = animate_nodes(graph, node_colors_r, scalarmappaple, colormap)
animation.save('gifs/average_500.gif', writer='imagemagick', savefig_kwargs={'facecolor':'white'}, fps=1)