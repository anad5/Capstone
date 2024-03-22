
# average run time for 100 time steps:
# health plots -> 3 mins
# gif -> 6 mins 
# can remove initial conditions of number of balls after a certain amount of timesteps to see if there is consensus 
#in pull ball, if time step is equal to memory than remove initial???
# need to check health plotting and superurn 

# memory 
# history -> deque max length which is the max memory, holds pairs of integer values 

import numpy as np
import matplotlib.pyplot as plt; plt.close('all')
import networkx as nx
from matplotlib.animation import FuncAnimation
import random as rd
import matplotlib.cm as cm
import matplotlib.colors as mcolors

from util import generate_graph, animate_nodes, update_super, init_urns, pull_ball, plot_health, plot_multi_health, plot_health_variance
from graph_dataALL import combine_graphs
from heuristic_functions import calc_centrality, calc_degree, calc_susceptibility, score_midpoint

# folder with all edge files 
circle_files_path = 'circles'

# combined graph file with edges
combined_file_path = 'circles/facebook_combined.txt'

# combine all graphs into one
graph = combine_graphs(circle_files_path, combined_file_path)

# count the number of nodes in the graph to double check all circles are being brought in 
number_of_nodes = graph.number_of_nodes()
print("Number of nodes in the combined graph:", number_of_nodes)

memory_flag = True
graph.graph['memory_flag'] = memory_flag

time_steps = 10
delta_red = 1
delta_blue = 1
init_red = 10
init_blue = 10
num_nodes = graph.number_of_nodes()

#memory_list = np.random.default_rng().normal(9, 3, size=num_nodes).astype(int)
#memory_list = np.where(memory_list<=0, 1, memory_list)

# memory list for old and young people
memory_list = np.zeros(num_nodes, dtype=int)
half_point = num_nodes // 2
memory_list[:half_point] = 6
memory_list[half_point:] = 3

init_red_list = np.random.default_rng().normal(10, 2, size=num_nodes).astype(int)
init_red_list = np.where(init_red_list<=0, 1, init_red_list)

init_blue_list = np.random.default_rng().normal(10, 2, size=num_nodes).astype(int)
init_blue_list = np.where(init_blue_list<=0, 1, init_blue_list)

init_urns(graph, init_red, init_blue, memory_list=memory_list, init_blue_list=init_blue_list, init_red_list=init_red_list)

# node IDs to indices 
node_to_index = {node: i for i, node in enumerate(graph.nodes())}

# calculate degree and centrality outside of time step loop 
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

# susceptability is calculated every time step, degree and centrality arent

all_scores = {} 
beta = 1
gamma = 1
alpha = 1

#with open('nodes_midpoint.txt', 'w') as f:
for i in range(time_steps):
    update_super(graph)
    pull_ball(graph, delta_blue, delta_red, i)
    suscept_score = {}
    for node in graph.nodes():
        node_idx = node_to_index[node]
        score = calc_susceptibility(graph, node, 'red', 'total')
        suscept_score[node_idx] = score
        #susceptibility_score[i] = all_score 
        
        # calculating heuristic scores 

    all_scores[i] = {}
    for node in graph.nodes():
        node_idx = node_to_index[node]
        degree_score = deg_score[node_idx] 
        centrality_score = central_score[node_idx]  
        susceptibility_score = suscept_score[node_idx]
            
        # Calculate the combined score using the formula from the screenshot
        combined = beta * degree_score + gamma * centrality_score - alpha * susceptibility_score
        all_scores[i][node_idx] = combined

        nodes_midpoint = score_midpoint(all_scores[i])

# need to add injected balls to the dequeue, add to the first value in pair 
    budget_red = 500
    delta_red = 1
    for node_idx in nodes_midpoint:
        if budget_red > 0:
            #graph.nodes[i]['history'][-1][0]+= delta_red 
            graph.nodes[node_idx]['red'] += delta_red
            graph.nodes[node_idx]['total'] += delta_red
            budget_red -= delta_red 
        else:
            break 

    budget_blue = 500
    #delta_blue = budget_blue/num_nodes
    for node in graph.nodes(): 
        #graph.nodes[i]['history'][-1][0]+= delta_blue
        graph.nodes[node_idx]['blue'] += delta_blue
        graph.nodes[node_idx]['total'] += delta_blue
        budget_blue -= delta_blue

health = np.empty((num_nodes, time_steps+1))
#for node in range(num_nodes):
for node in graph.nodes():
    index = node_to_index[node]  # Convert node ID to index
    health[index, :] = graph.nodes[node]['health']
    #health[node] = graph.nodes[node]['health']

health = np.array(health)

plot_health(graph, health)
plot_multi_health(graph, health)
plot_health_variance(graph, health)

# node IDs to indices 
node_to_index = {node: i for i, node in enumerate(graph.nodes())}

# Initialize the health array based on the number of nodes
num_nodes = len(graph.nodes())
health = np.empty((num_nodes, time_steps+1))

# Use the mapping when accessing the health array
for node in graph.nodes():
    index = node_to_index[node]  # Convert node ID to index
    health[index, :] = graph.nodes[node]['health']

node_colors_r = health[:, :-1].T  # Use slicing to exclude the last column for all rows
node_colors_r_1 = health[:, :-1]  # This is the same as node_colors_r without transposition

# Assuming node_colors_template is to be used for some other purpose and unrelated to the node-to-index mapping
node_colors_template = np.random.randint(0, 100, size=(time_steps, num_nodes))

node_colors_test = np.ones((time_steps, num_nodes))

normalize = mcolors.Normalize(vmin=0, vmax=1)
colormap = mcolors.LinearSegmentedColormap.from_list("MyCmapName",["b","r"])
colormap = plt.colormaps.get_cmap('seismic')

scalarmappaple = cm.ScalarMappable(norm=normalize, cmap=colormap)
scalarmappaple.set_array(health[0,:])

animation = animate_nodes(graph, node_colors_r, scalarmappaple, colormap)
animation.save('gifs/redAverage_500_oldyoung.gif', writer='imagemagick', savefig_kwargs={'facecolor':'white'}, fps=1)