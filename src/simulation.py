import numpy as np
import matplotlib.pyplot as plt; plt.close('all')
import networkx as nx
from matplotlib.animation import FuncAnimation
import random as rd
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from util import animate_nodes, update_super, pull_ball, init_urns, generate_graph, pyvis_animation, get_degrees

#graph = nx.complete_graph(total_nodes)
graph = generate_graph("./src/graph_data/Fig5_1_c_Adjacency_Matrix.txt")
time_steps = 10
delta_red = 1
delta_blue = 1
init_red=2
init_blue=2
num_nodes = graph.number_of_nodes()

init_urns(graph, init_red, init_blue)

for i in range(time_steps):
    update_super(graph)
    pull_ball(graph, delta_red, delta_blue)

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
colormap = cm.get_cmap('seismic')

scalarmappaple = cm.ScalarMappable(norm=normalize, cmap=colormap)
scalarmappaple.set_array(health[0,:])


pyvis_animation(graph)
#animation = animate_nodes(graph, node_colors_r, scalarmappaple, colormap)
#animation.save('test2.gif', writer='imagemagick', savefig_kwargs={'facecolor':'white'}, fps=1)
