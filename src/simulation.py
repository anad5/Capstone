import numpy as np
import matplotlib.pyplot as plt; plt.close('all')
import networkx as nx
from matplotlib.animation import FuncAnimation
import random as rd
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from util import animate_nodes, update_super, pull_ball, init_urns, generate_graph, pyvis_animation, get_scores, plot_health, inject_uniform_red, inject_relative_red, combine_graphs, animate_nodes_lae, inject_uniform_blue, plot_health_variance

#graph = nx.complete_graph(total_nodes)
circles_flag = True
memory_flag = True
remove_init_flag = True

if circles_flag:
    circle_files_path = 'circles'

    # combined graph file with edges
    combined_file_path = 'circles/facebook_combined.txt'

    # combine all graphs into one
    graph = combine_graphs(circle_files_path, combined_file_path)
    graph.graph['memory_flag'] = memory_flag
    graph.graph['remove_init_flag'] = remove_init_flag

    # count the number of nodes in the graph to double check all circles are being brought in 
    num_nodes = graph.number_of_nodes()
    print("Number of nodes in the combined graph:", num_nodes)
else:
    graph = generate_graph("./src/graph_data/Fig5_1_c_Adjacency_Matrix.txt")
    num_nodes = graph.number_of_nodes()
    print("Number of nodes in the combined graph:", num_nodes)


time_steps = 100
delta_red = 2
delta_blue = 2
init_red = 10
init_blue = 10
budget = 500
budget_b = 500
alpha = 10
beta = 1
gamma = 1


memory_list = np.random.default_rng().normal(9, 3, size=num_nodes).astype(int)
memory_list = np.where(memory_list<=0, 1, memory_list)

init_red_list = np.random.default_rng().normal(10, 2, size=num_nodes).astype(int)
init_red_list = np.where(init_red_list<=0, 1, init_red_list)

init_blue_list = np.random.default_rng().normal(10, 2, size=num_nodes).astype(int)
init_blue_list = np.where(init_blue_list<=0, 1, init_blue_list)

init_urns(graph, init_red, init_blue, memory_list=memory_list, init_blue_list=init_blue_list, init_red_list=init_red_list)

closeness = nx.closeness_centrality(graph) # Calculate closeness once and then just pass as arg to score function

for i in range(time_steps):
    update_super(graph)
    scores = get_scores(graph, i, closeness, alpha, beta, gamma, quantize=False)
    if i < 1:
        print("Test")
    else:
        inject_relative_red(graph, scores, budget)
        inject_uniform_blue(graph, scores, budget)    
    pull_ball(graph, delta_red, delta_blue, i, init_red_list, init_blue_list)

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

#normalize = mcolors.Normalize(vmin=health.min(), vmax=health.max())gith
normalize = mcolors.Normalize(vmin=0, vmax=1)
#colormap = cm.jet
colormap = mcolors.LinearSegmentedColormap.from_list("MyCmapName",["b","r"])
colormap = plt.colormaps.get_cmap('seismic')

scalarmappaple = cm.ScalarMappable(norm=normalize, cmap=colormap)
scalarmappaple.set_array(health[0,:])

#plot_health(graph, health, alpha, beta, gamma)
plot_health_variance(graph, health, alpha, beta, gamma)

animation = animate_nodes_lae(graph, node_colors_r, scalarmappaple, colormap)
animation.save(f"gifs/sim_alpha{alpha}_beta{beta}_gamma{gamma}.gif", writer='imagemagick', savefig_kwargs={'facecolor':'white'}, fps=1)

test = 1
"""
health = np.empty((num_nodes, time_steps+1))
for node in range(num_nodes):
    health[node] = graph.nodes[node]['health']

health = np.array(health)

plot_health(graph, health)

node_colors_r = health[:,:-1].T
node_colors_r_1 = health[:,:-1]
node_colors_template = np.random.randint(0, 100, size=(time_steps, num_nodes))


node_colors_test = np.ones((time_steps, num_nodes))

#normalize = mcolors.Normalize(vmin=health.min(), vmax=health.max())
normalize = mcolors.Normalize(vmin=0, vmax=1)
#colormap = cm.jet
colormap = mcolors.LinearSegmentedColormap.from_list("MyCmapName",["b","r"])
colormap = cm.get_cmap('seismic')

scalarmappaple = cm.ScalarMappable(norm=normalize, cmap=colormap)
scalarmappaple.set_array(health[0,:])


#pyvis_animation(graph)
animation = animate_nodes(graph, node_colors_r, scalarmappaple, colormap)
animation.save('test2.gif', writer='imagemagick', savefig_kwargs={'facecolor':'white'}, fps=1)"""
