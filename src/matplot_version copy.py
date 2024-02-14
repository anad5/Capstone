import numpy as np
import matplotlib.pyplot as plt; plt.close('all')
import networkx as nx
from matplotlib.animation import FuncAnimation
import random as rd
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from util import generate_graph

# Added
def animate_nodes(G, node_colors, scalarmappaple, colormap, pos=None, *args, **kwargs):

    fig, ax = plt.subplots() 
    plt.title('Polya Urn Network')

    # define graph layout if None given
    if pos is None:
        pos = nx.spring_layout(G)

    # draw graph
    #plt.title('Polya Urn Network')
    #cbar = plt.colorbar(scalarmappaple)
    #cbar.set_label('Brand awareness')
        
    #initial
    nodes = nx.draw_networkx_nodes(G, pos, node_color=node_colors[0, :], node_size = 100, cmap=colormap, ax=ax, *args, **kwargs)
    edges = nx.draw_networkx_edges(G, pos, ax=ax, *args, **kwargs)

    scalarmappaple.set_array(node_colors[0, :])

    cbar = fig.colorbar(scalarmappaple, ax=ax)  # Specify the ax argument
    cbar.set_label('Brand awareness', fontsize=12)

    #rgba_array_i = scalarmappaple.to_rgba(node_colors[0,:])
    #nodes = nx.draw_networkx_nodes(G, pos, node_color=rgba_array_i, cmap=colormap , *args, **kwargs)
    #nodes = nx.draw_networkx_nodes(G, pos, cmap=colormap , *args, **kwargs)
    #edges = nx.draw_networkx_edges(G, pos, *args, **kwargs)
    #nodes.set_cmap(colormap)
    #plt.axis('off')

    #nodes.set_array(node_colors[0])
    def update(ii):
        # nodes are just markers returned by plt.scatter;
        # node color can hence be changed in the same way like marker colors\
        rgba_array = scalarmappaple.to_rgba(node_colors[ii,:])
        nodes.set_color(rgba_array)
        #nodes = nx.draw_networkx_nodes(G, pos, node_color=rgba_array, cmap=colormap , *args, **kwargs)
        #test1 = np.expand_dims(test, axis=1)
        #test2 = np.broadcast_to(test1, (test1.shape[0], 4))
        #nodes.set_facecolor(test2)
        #nodes.set_array(test)
        return nodes,

    #fig = plt.gcf()
    frames=len(node_colors[:,0])
    #animation = FuncAnimation(fig, update, interval=50, frames=len(node_colors[:,0]), blit=True)
    animation = FuncAnimation(fig, update, frames=frames, blit=True)
    plt.close()
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

# Added
def pull_ball(graph):
    num_nodes = graph.number_of_nodes()
    for node in range(num_nodes):
        random_pull = rd.uniform(0,1)
        threshold = graph.nodes[node]['super_red']/graph.nodes[node]['super_total']
        if random_pull < threshold: # Pulled a red ball
            graph.nodes[node]['red'] += delta_red
            graph.nodes[node]['total'] += delta_red
        else:
            graph.nodes[node]['blue'] += delta_blue
            graph.nodes[node]['total'] += delta_blue
        graph.nodes[node]['health'].append((graph.nodes[node]['red']/graph.nodes[node]['total'])) # Update the health of each node
        #graph.nodes[node]['health'].append(int((graph.nodes[node]['red']/graph.nodes[node]['total'])*100)) # Update the health of each node

# Added
def init_urns(graph):
    num_nodes = graph.number_of_nodes()
    for node in range(num_nodes):
        # i, red=2, blue=2, total=4, super_red=2, super_blue=2, super_total=4, health=[1], pos=(i,1)
        graph.nodes[node]['red'] = 2
        graph.nodes[node]['blue'] = 2
        graph.nodes[node]['total'] = 4
        graph.nodes[node]['super_red'] = 2
        graph.nodes[node]['super_blue'] = 2
        graph.nodes[node]['super_total'] = 2
        graph.nodes[node]['health'] = [0.5]

#graph = nx.complete_graph(total_nodes)
graph = generate_graph("./src/graph_data/Fig5_1_c_Adjacency_Matrix.txt")
time_steps = 40
delta_red = 1
delta_blue = 1
num_iters = 10
num_nodes = graph.number_of_nodes()

init_urns(graph)

for i in range(time_steps):
    update_super(graph)
    pull_ball(graph)

health = np.empty((num_nodes, time_steps+1))
for node in range(num_nodes):
    health[node] = graph.nodes[node]['health']

health = np.array(health)

node_colors_r = health[:,:-1].T
node_colors_r_1 = health[:,:-1]
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
animation.save('test.gif', writer='imagemagick', savefig_kwargs={'facecolor':'white'}, fps=1)