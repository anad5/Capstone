import matplotlib.pyplot as plt; plt.close('all')
import networkx as nx
from matplotlib.animation import FuncAnimation
import random as rd
import numpy as np

def animate_nodes(G, node_colors, scalarmappaple, colormap, pos=None, *args, **kwargs):
    """
    Function to animate node colours changing.

    Args:
        G: Networkx graph structure.
        node_colors: 2D Numpy array of urn health over time. Time in dim 0. Urns in dim 1. TODO: Check dims
        scalarmappaple: matplotlib color code TODO: Rewrite this with more accurate info
        colormap: TODO: What is this?
        pos: Default is None TODO: What is this?

        TODO: Finish writing this

    Returns:
        Animation object from matplotlib.animation.

    Raises:
        None.
    """
    # define graph layout if None given
    if pos is None:
        pos = nx.spring_layout(G)

    # draw graph
    plt.title('Polya Urn Network')
    cbar = plt.colorbar(scalarmappaple)
    cbar.set_label('Brand awareness')

    rgba_array_i = scalarmappaple.to_rgba(node_colors[0,:])
    #nodes = nx.draw_networkx_nodes(G, pos, node_color=rgba_array_i, cmap=colormap , *args, **kwargs)
    nodes = nx.draw_networkx_nodes(G, pos, cmap=colormap , *args, **kwargs)
    edges = nx.draw_networkx_edges(G, pos, *args, **kwargs)
    nodes.set_cmap(colormap)
    #plt.axis('off')

    #nodes.set_array(node_colors[0])
    def update(ii):
        # nodes are just markers returned by plt.scatter;
        # node color can hence be changed in the same way like marker colors\
        rgba_array = scalarmappaple.to_rgba(node_colors[ii,:])
        nodes = nx.draw_networkx_nodes(G, pos, node_color=rgba_array, cmap=colormap , *args, **kwargs)
        #test1 = np.expand_dims(test, axis=1)
        #test2 = np.broadcast_to(test1, (test1.shape[0], 4))
        #nodes.set_facecolor(test2)
        #nodes.set_array(test)
        return nodes,

    fig = plt.gcf()
    frames=len(node_colors[:,0])
    animation = FuncAnimation(fig, update, interval=50, frames=len(node_colors[:,0]), blit=True)
    return animation

def update_super(graph):
    """
    Function to update super urns after pulling balls.

    Args:
        graph: Networkx graph structure.

    Returns:
        None.

    Raises:
        None.
    """
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

def pull_ball(graph, delta_red, delta_blue):
    """
    Function to simulate pulling of a ball from all urns. Updates only local urns.

    Args:
        graph: Networkx graph structure.
        delta_red: Number of red balls to be added on red pull.
        delta_blue: Number of blue balls to be added on blue pull.

    Returns:
        None.

    Raises:
        None.
    """
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

def init_urns(graph, init_red, init_blue):
    """
    Function to initialize red and blue balls in urns and super urns.

    Args:
        graph: Networkx graph object to be initialized.
        init_red: Initial number of red balls.
        init_blue: Initial number of blue balls.

    Returns:
        None.

    Raises:
        None.
    """
    num_nodes = graph.number_of_nodes()
    for node in range(num_nodes):
        # i, red=2, blue=2, total=4, super_red=2, super_blue=2, super_total=4, health=[1], pos=(i,1)
        graph.nodes[node]['red'] = init_red
        graph.nodes[node]['blue'] = init_blue
        graph.nodes[node]['total'] = init_red + init_blue
        graph.nodes[node]['super_red'] = init_red
        graph.nodes[node]['super_blue'] = init_blue
        graph.nodes[node]['super_total'] = init_red + init_blue
        graph.nodes[node]['health'] = [init_red/(init_blue+init_red)]

def generate_graph(adj_matrix_path, skiprows=0):
    """
    Function to generate networkx graph object from adjacency matrix.

    Args:
        adj_matrix_path: Path to adjacency matrix file.
        skiprows: Number of rows to skip in adj matrix file - default it 0.

    Returns:
        Returns networkx graph object.

    Raises:
        None.
    """
    np_array = np.loadtxt(open(adj_matrix_path, "rb"), delimiter=",", skiprows=skiprows)
    graph = nx.from_numpy_array(np_array)
    return graph