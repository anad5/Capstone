import matplotlib.pyplot as plt; plt.close('all')
import networkx as nx
from matplotlib.animation import FuncAnimation
from matplotlib.colors import Normalize
import random as rd
import numpy as np
from collections import deque
from pyvis.network import Network
from datetime import datetime
import os

plt.ioff() # Turn off interactive plotting

def combine_graphs(circle_files_directory, combined_file=None):
    # create G empty graph 
    G = nx.Graph()
    
    # combined edges file 
    if combined_file:
        with open(combined_file, 'r') as f:
            for line in f:
                try:
                    parts = line.strip().split()
                    if len(parts) == 2:
                        src, dst = map(int, parts)
                        G.add_edge(src, dst)
                except ValueError:
                    # I DONT KNOW WHY THESE DON'T WORK NEED TO FIGURE OUT
                    print(f"Skipping line with invalid integers: {line.strip()}")

    # adding the edges from individual circle edge file 
    for edge_file_name in os.listdir(circle_files_directory):
        edge_file_path = os.path.join(circle_files_directory, edge_file_name)
        with open(edge_file_path, 'r') as f:
            for line in f:
                try:
                    parts = line.strip().split()
                    if len(parts) == 2:
                        src, dst = map(int, parts)
                        G.add_edge(src, dst)
                except ValueError:
                    # CHECK THIS 
                    print(f"Skipping line with invalid integers: {line.strip()}")

    return G

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
        pos = nx.spring_layout(G, k = 0.08)

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

def animate_nodes_lae(graph, node_colors, scalarmappaple, colormap, pos=None, node_size=8, *args, **kwargs):
    fig, ax = plt.subplots() 
    plt.title('Polya Urn Network')

    # Define graph layout if None given
    if pos is None:
        pos = nx.spring_layout(graph, k=0.07)

    # Draw edges
    nx.draw_networkx_edges(graph, pos, width=0.25, ax=ax, *args, **kwargs)

    # Extract node positions
    node_x = [pos[node][0] for node in graph.nodes()]
    node_y = [pos[node][1] for node in graph.nodes()]

    # Normalize colors
    norm = Normalize(vmin=min(node_colors[0]), vmax=max(node_colors[0]))

    # Compute node colors
    rgba_colors = [colormap(norm(color)) for color in node_colors[0]]

    # Draw nodes with custom colors and size
    nodes = ax.scatter(node_x, node_y, c=rgba_colors, s=node_size)

    # Set color bar
    scalarmappaple.set_array(node_colors[0])
    cbar = fig.colorbar(scalarmappaple, ax=ax)
    cbar.set_label('Brand awareness', fontsize=12)

    # Update function to change node colors
    def update(ii):
        rgba_colors = [colormap(norm(color)) for color in node_colors[ii]]
        nodes.set_color(rgba_colors)
        return nodes,

    frames = len(node_colors)
    animation = FuncAnimation(fig, update, frames=frames, blit=True)
    plt.close()
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
        # Add contents of local urn to super urn
        graph.nodes[node]['super_red'] = graph.nodes[node]['red']
        graph.nodes[node]['super_blue'] = graph.nodes[node]['blue']
        graph.nodes[node]['super_total'] = graph.nodes[node]['red'] + graph.nodes[node]['blue']
        # Add contents of neighbours local urns to super urn
        for neighbor in graph.neighbors(node):
            red = graph.nodes[neighbor]['red']
            blue = graph.nodes[neighbor]['blue']
            graph.nodes[node]['super_red'] += red
            graph.nodes[node]['super_blue'] += blue
            graph.nodes[node]['super_total'] += red + blue

def pull_ball(graph, delta_red, delta_blue, i, init_red_list, init_blue_list):
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

        # Remove balls that have expired
        if graph.graph['memory_flag'] == True:
            if len(graph.nodes[node]['history']) == graph.nodes[node]['memory']:
                disapearing = graph.nodes[node]['history'].popleft()
                #testing#
                graph.nodes[node]['red'] -= disapearing[0]
                graph.nodes[node]['blue'] -= disapearing[1]
                graph.nodes[node]['total'] -= sum(disapearing)

        # Remove initial conditions
        if i == graph.nodes[node]['memory'] and graph.graph['remove_init_flag'] == True:
                graph.nodes[node]['red'] -= init_red_list[node]
                graph.nodes[node]['blue'] -= init_blue_list[node]
                graph.nodes[node]['total'] -= (init_red_list[node] + init_blue_list[node])

        # Pulling process
        random_pull = rd.uniform(0,1)
        threshold = graph.nodes[node]['super_red']/graph.nodes[node]['super_total']
        if random_pull < threshold: # Pulled a red ball
            graph.nodes[node]['red'] += delta_red
            graph.nodes[node]['total'] += delta_red
            if graph.graph['memory_flag'] == True:
                graph.nodes[node]['history'].append([1,0]) # Add red ball indicator to history
        else:
            graph.nodes[node]['blue'] += delta_blue
            graph.nodes[node]['total'] += delta_blue
            if graph.graph['memory_flag'] == True:
                graph.nodes[node]['history'].append([0,1]) # Add blue ball indicator to history

        # Update the health of each node
        graph.nodes[node]['health'].append((graph.nodes[node]['super_red']/graph.nodes[node]['super_total'])) 
        

def init_urns(graph, init_red, init_blue, memory=5, memory_list=None, init_blue_list=None, init_red_list=None):
    """
    Function to initialize red and blue balls in urns and super urns.

    Args:
        graph: Networkx graph object to be initialized.
        init_red: Initial number of red balls.
        init_blue: Initial number of blue balls.
        memory: Value for uniform memory length. Default is 5. 
        memory_list: Optional list for memories of urns - in oder of nodes in graph. If no list passed then memories are defaulted to memory value.

    Returns:
        None.

    Raises:
        None.
    """
    num_nodes = graph.number_of_nodes()
    for node in range(num_nodes):
        if init_blue_list.any():
            init_blue=init_blue_list[node]
        if init_red_list.any():
            init_red = init_red_list[node]
        # i, red=2, blue=2, total=4, super_red=2, super_blue=2, super_total=4, health=[1], pos=(i,1)
        graph.nodes[node]['red'] = init_red
        graph.nodes[node]['blue'] = init_blue
        graph.nodes[node]['total'] = init_red + init_blue
        graph.nodes[node]['super_red'] = init_red
        graph.nodes[node]['super_blue'] = init_blue
        graph.nodes[node]['super_total'] = init_red + init_blue
        graph.nodes[node]['health'] = [init_red/(init_blue+init_red)]
        if graph.graph['memory_flag'] == True:
            if memory_list.any(): # Distinct memory values for urns
                graph.nodes[node]['memory'] = memory_list[node]
            else: # Uniform memory distribution
                graph.nodes[node]['memory'] = memory
            graph.nodes[node]['history'] = deque(maxlen=int(graph.nodes[node]['memory'])) # Deque to store information on which balls to be removed at what times
            
def generate_graph(adj_matrix_path, memory_flag=True, remove_init_flag=True, skiprows=0):
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
    G = nx.from_numpy_array(np_array)
    G.graph['memory_flag'] = memory_flag
    G.graph['remove_init_flag'] = remove_init_flag
    return G

def heuristic(degree, centrality, susceptibility, memory, alpha, beta, gamma, zeta):
    return beta*degree+gamma*centrality-alpha*susceptibility+zeta*memory

def quantize_score(score, levels=[0,10,20,30,40,50,60]):
    for level in range(levels):
        if score<levels[level+1]:
            score = levels[level]
            return score
    return levels[-1]

def get_scores(G, i, closeness, alpha, beta, gamma, zeta, quantize=True):
    """
    Function to calculate heuristic scores for each node in a graph.

    Args:
        G: networkx graph structure
        i: time index to get health values.

    Returns:
        4xn numpy array with attributes and scores - scores in row 4.

    Raises:
        None.
    """
    # Can return these as dictionaries or as numpy arrays
    # Reference these metrics for choice of centrality https://networkx.org/documentation/stable/reference/algorithms/centrality.html
    # Dictionary with closeness values, index by node number
    #katz = nx.katz_centrality_numpy(G, alpha=0.1) # Numpy array with katz centrality values
    #eigen_centrality = nx.eigenvector_centrality(G) # Dictionary with closeness values, index by node number

    num_nodes = G.number_of_nodes()
    scores = np.empty((4, num_nodes))
    for node in range(num_nodes):
        degree = G.degree[node]
        centrality = closeness[node] # TODO: Choose centrality metric
        susceptibility = G.nodes[node]['health'][i] # TODO: Health is currently a list not an int, choose most recent or add new attribute of only most recent
        scores[0,node] = degree
        scores[1,node] = centrality
        scores[2,node] = susceptibility
        memory = G.nodes[node]['memory']
        if quantize:
            score = int(heuristic(degree, centrality, susceptibility, memory, alpha, beta, gamma, zeta))
            quantized_score = quantize_score(score)
            scores[3,node] = quantized_score
        else:
            scores[3,node] = int(heuristic(degree, centrality, susceptibility, memory, alpha, beta, gamma, zeta))
    return scores

def inject_uniform_red(G, scores, budget_b):
    """
    Function to inject red balls to urns with the top n scores uniformly

    Args:
        graph: Networkx graph structure.
        health: Array of health values at each timestep.

    Returns:
        None.

    Raises:
        None.
    """
    num = len(scores[3,:])
    num = G.number_of_nodes()
    for i in range(num):
        if G.graph['memory_flag'] == True:
            G.nodes[i]['history'][-1][0] += budget_b / num 
            G.nodes[i]['red'] += budget_b / num
        else:
            G.nodes[i]['red'] += budget_b / num
        G.nodes[i]['total'] += budget_b / num

def inject_uniform_blue(G, scores, budget_b):
    """
    Function to inject red balls to urns with the top n scores uniformly

    Args:
        graph: Networkx graph structure.
        health: Array of health values at each timestep.

    Returns:
        None.

    Raises:
        None.
    """
    num = len(scores[3,:])
    num = G.number_of_nodes()
    for i in range(num):
        if G.graph['memory_flag'] == True:
            G.nodes[i]['history'][-1][1] += budget_b / num 
            G.nodes[i]['blue'] += budget_b / num
        else:
            G.nodes[i]['blue'] += budget_b / num
        G.nodes[i]['total'] += budget_b / num

def inject_relative_red(G, scores, budget):
    """
    Function to inject red balls to urns with the top n scores relative to their scores

    Args:
        G: Networkx graph structure.
        scores: Array of heuristic scores.

    Returns:
        None.

    Raises:
        None.
    """
    total = np.sum(scores[3, :])
    for i in range(len(scores[3, :])):
        relative = scores[3, i] / total
        if G.graph['memory_flag'] == True:
            amount = budget * relative
            G.nodes[i]['history'][-1][0] += amount
            G.nodes[i]['red'] += amount
            G.nodes[i]['total'] += amount
        else:
            G.nodes[i]['red'] += budget * relative
            G.nodes[i]['total'] += amount

def plot_health_variance(G, health, alpha=1, beta=1, gamma=1):
    """
    Function to plot the overall "health" of the network over all timesteps

    Args:
        graph: Networkx graph structure.
        health: Array of health values at each timestep.

    Returns:
        None.

    Raises:
        None.
    """
    avg_health = np.var(health[:,:-1]*100, axis=0)
    plt.plot(avg_health)
    plt.xlabel('Timestep')
    plt.ylabel(f"Variance of Network Exposure - Alpha: {alpha}, Beta: {beta}, Gamma: {gamma}")

    plt.savefig(f"./figures/health_variance_plot_alpha{alpha}_beta{beta}_gamma{gamma}.png")
    plt.clf()

def plot_multi_health(G, health, alpha, beta, gamma):
    """
    Function to plot the "health" of a few random nodes over all timesteps

    Args:
        graph: Networkx graph structure.
        health: Array of health values at each timestep.

    Returns:
        None.

    Raises:
        None.
    """
    num_nodes = G.number_of_nodes()
    nums = rd.sample(range(0, num_nodes), 4)
    multi_health = health[nums,:-1]
    plt.plot(multi_health.T)
    plt.legend(['Node 1', 'Node 2', 'Node 3', 'Node 4'])
    plt.xlabel('Timestep')
    plt.ylabel(f"Network Exposure - Alpha: {alpha}, Beta: {beta}, Gamma: {gamma}")
    ax = plt.gca()
    #giax.set_ylim([0, 1])
    plt.savefig(f"./figures/multi_health_plot_alpha{alpha}_beta{beta}_gamma{gamma}.png")
    plt.clf()

def plot_health(G, health, alpha=1, beta=1, gamma=1):
    """
    Function to plot the overall "health" of the network over all timesteps

    Args:
        graph: Networkx graph structure.
        health: Array of health values at each timestep.

    Returns:
        None.

    Raises:
        None.
    """
    num_nodes = G.number_of_nodes()
    avg_health = np.sum(health[:,:-1], axis=0)/num_nodes
    plt.plot(avg_health)
    plt.xlabel('Timestep')
    plt.ylabel(f"Average Network Exposure - Alpha: {alpha}, Beta: {beta}, Gamma: {gamma}")

    plt.savefig(f"./figures/health_plot_alpha{alpha}_beta{beta}_gamma{gamma}.png")
    plt.clf()


def pyvis_animation(G, width='500px', height='500px'):
    """
    Function to animate node colours changing using pyvis.

    Args:
        G: Networkx graph structure.
        width: Width of animation box in pixels e.g. '500px'
        width: Height of animation box in pixels e.g. '500px'


    Returns:
        None.

    Raises:
        None.
    """
    for node in range(G.number_of_nodes()):
        G.nodes[node].pop('history',None)
    nt = Network(width, height)
    nt.from_nx(G)
    nt.show('nx.html', notebook=False)