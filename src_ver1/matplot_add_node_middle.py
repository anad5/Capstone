import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.animation import FuncAnimation
import random as rd
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from util import generate_graph  # Assuming this import is correct and available

# Added
def animate_nodes(G, node_colors, scalarmappaple, colormap, pos=None, *args, **kwargs):
    fig, ax = plt.subplots()
    plt.title('Polya Urn Network')

    # define graph layout if None given
    if pos is None:
        pos = nx.spring_layout(G)

    # draw graph
    nodes = nx.draw_networkx_nodes(G, pos, node_color=node_colors[0, :], node_size=100, cmap=colormap, ax=ax, *args, **kwargs)
    edges = nx.draw_networkx_edges(G, pos, ax=ax, *args, **kwargs)

    scalarmappaple.set_array(node_colors[0, :])

    cbar = fig.colorbar(scalarmappaple, ax=ax)  # Specify the ax argument
    cbar.set_label('Brand awareness', fontsize=12)

    def update(ii):
        rgba_array = scalarmappaple.to_rgba(node_colors[ii, :])
        nodes.set_array(node_colors[ii, :])
        nodes.set_color(rgba_array)
        scalarmappaple.set_array(node_colors[ii, :])
        cbar.update_normal(scalarmappaple)
        return nodes,

    frames = len(node_colors[:, 0])
    animation = FuncAnimation(fig, update, frames=frames, blit=True)
    plt.close()  # Close the initial figure to prevent extra plots
    return animation

# Added
def update_super(graph):
    num_nodes = graph.number_of_nodes()
    for node in range(num_nodes):
        graph.nodes[node]['super_red'] = graph.nodes[node]['red']
        graph.nodes[node]['super_blue'] = graph.nodes[node]['blue']
        graph.nodes[node]['super_total'] = graph.nodes[node]['red'] + graph.nodes[node]['blue']
        for neighbor in graph.neighbors(node):
            red = graph.nodes[neighbor]['red']
            blue = graph.nodes[neighbor]['blue']
            graph.nodes[node]['super_red'] += red
            graph.nodes[node]['super_blue'] += blue
            graph.nodes[node]['super_total'] += red + blue

# Updated
def pull_ball(graph, influencer_probability=0.1):
    num_nodes = graph.number_of_nodes()
    for node in range(num_nodes):
        random_pull = rd.uniform(0, 1)
        if node == influencer_node:
            # The influencer node always pulls a red ball
            graph.nodes[node]['red'] += delta_red
            graph.nodes[node]['total'] += delta_red
        elif random_pull < graph.nodes[node]['super_red'] / graph.nodes[node]['super_total']:
            graph.nodes[node]['red'] += delta_red
            graph.nodes[node]['total'] += delta_red
        else:
            graph.nodes[node]['blue'] += delta_blue
            graph.nodes[node]['total'] += delta_blue
        graph.nodes[node]['health'].append((graph.nodes[node]['red'] / graph.nodes[node]['total']))

# Updated
def init_urns(graph, influencer_node):
    num_nodes = graph.number_of_nodes()
    for node in range(num_nodes):
        graph.nodes[node]['red'] = 2
        graph.nodes[node]['blue'] = 2
        graph.nodes[node]['total'] = 4
        graph.nodes[node]['super_red'] = 2
        graph.nodes[node]['super_blue'] = 2
        graph.nodes[node]['super_total'] = 2
        graph.nodes[node]['health'] = [0.5]

    # Initialize influencer node
    graph.nodes[influencer_node]['red'] = 4  # Influencer starts with all red balls
    graph.nodes[influencer_node]['blue'] = 0
    graph.nodes[influencer_node]['total'] = 4
    graph.nodes[influencer_node]['super_red'] = 4
    graph.nodes[influencer_node]['super_blue'] = 0
    graph.nodes[influencer_node]['super_total'] = 4
    graph.nodes[influencer_node]['health'] = [1.0]  # Influencer starts with all red balls

    # Connect influencer node to existing nodes - connects it to all nodes.
    for node in range(num_nodes):
        graph.add_edge(influencer_node, node)

# Example Usage
graph = generate_graph("./src_ver1/graph_data/Fig5_1_c_Adjacency_Matrix.txt")
time_steps = 40
delta_red = 1
delta_blue = 1
num_nodes = graph.number_of_nodes()
influencer_node = 0  # Set influencer_node to a valid index (e.g., 0 for the first node)

init_urns(graph, influencer_node)

for i in range(time_steps):
    update_super(graph)
    pull_ball(graph)

health = np.empty((num_nodes, time_steps+1))
for node in range(num_nodes):
    health[node] = graph.nodes[node]['health']

health = np.array(health)

node_colors_r = health[:, :-1].T

normalize = mcolors.Normalize(vmin=0, vmax=1)
colormap = plt.colormaps.get_cmap('seismic')

scalarmappaple = cm.ScalarMappable(norm=normalize, cmap=colormap)
scalarmappaple.set_array(health[0, :])

animation = animate_nodes(graph, node_colors_r, scalarmappaple, colormap)
animation.save('Test_Influencer_Middle_1.gif', writer='imagemagick', savefig_kwargs={'facecolor':'white'}, fps=1)

plt.show()  # To display the final plot
