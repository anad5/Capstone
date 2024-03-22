from collections import deque
import plotly.graph_objects as go
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import random as rd
from matplotlib.colors import Normalize, ListedColormap


def generate_edge_trace(graph):
    edge_x = []
    edge_y = []

    for edge in graph.edges():
        test = graph.nodes[edge[0]]
        x0, y0 = graph.nodes[edge[0]]['pos']
        x1, y1 = graph.nodes[edge[1]]['pos']
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')
    return edge_trace

def generate_node_trace(graph):
    node_x = []
    node_y = []
    for node in graph.nodes():
        x, y = graph.nodes[node]['pos']
        node_x.append(x)
        node_y.append(y)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            # colorscale options
            #'Greys' | 'YlGnBu' | 'Greens' | 'YlOrRd' | 'Bluered' | 'RdBu' |
            #'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |
            #'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' |
            colorscale='YlGnBu',
            reversescale=True,
            color=[],
            size=10,
            colorbar=dict(
                thickness=15,
                title='Node Connections',
                xanchor='left',
                titleside='right'
            ),
            line_width=2))
    return node_trace

def generate_graph(adj_matrix_path, skiprows=0):
    np_array = np.loadtxt(open(adj_matrix_path, "rb"), delimiter=",", skiprows=skiprows)
    graph = nx.from_numpy_array(np_array)
    
    return graph

def animate_nodes(graph, node_colors, scalarmappaple, colormap, pos=None, node_size=8, *args, **kwargs):
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
    

# # checks if deque has reached memory and then it will pop the left most ball and will subtract balls
# def pull_ball(graph, delta_red, delta_blue):
#     """
#     Function to simulate pulling of a ball from all urns. Updates only local urns.

#     Args:
#         graph: Networkx graph structure.
#         delta_red: Number of red balls to be added on red pull.
#         delta_blue: Number of blue balls to be added on blue pull.

#     Returns:
#         None.

#     Raises:
#         None.
#     """
#     num_nodes = graph.number_of_nodes()
#     for node in range(num_nodes):
#         # Remove balls that have expired
#         if graph.graph['memory_flag'] == True:
#             if len(graph.nodes[node]['history']) == graph.nodes[node]['memory']:
#                 disapearing = graph.nodes[node]['history'].popleft()
#                 #testing#
#                 graph.nodes[node]['red'] -= disapearing[0]
#                 graph.nodes[node]['blue'] -= disapearing[1]
#                 graph.nodes[node]['total'] -= sum(disapearing)
#                 #testing#
#                 '''if disapearing > 0:
#                     graph.nodes[node]['red'] -= disapearing
#                 else:
#                     graph.nodes[node]['blue'] -= 1'''
#         random_pull = rd.uniform(0,1)
#         threshold = graph.nodes[node]['super_red']/graph.nodes[node]['super_total']
#         if random_pull < threshold: # Pulled a red ball
#             graph.nodes[node]['red'] += delta_red
#             graph.nodes[node]['total'] += delta_red
#             if graph.graph['memory_flag'] == True:
#                 graph.nodes[node]['history'].append([1,0]) # Add red ball indicator to history
#         else:
#             graph.nodes[node]['blue'] += delta_blue
#             graph.nodes[node]['total'] += delta_blue
#             if graph.graph['memory_flag'] == True:
#                 graph.nodes[node]['history'].append([0,1]) # Add blue ball indicator to history
#         graph.nodes[node]['health'].append((graph.nodes[node]['super_red']/graph.nodes[node]['super_total'])) # Update the health of each node
            
# pull ball that removes initial conditions
def pull_ball(graph, delta_red, delta_blue, current_step):
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
                #testing#
                '''if disapearing > 0:
                    graph.nodes[node]['red'] -= disapearing
                else:
                    graph.nodes[node]['blue'] -= 1'''
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
        graph.nodes[node]['health'].append((graph.nodes[node]['super_red']/graph.nodes[node]['super_total'])) # Update the health of each node


def plot_health(G, health):
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
    plt.ylabel('Average Network Exposure')
    
    plt.savefig(f"health/redAverage_500_OVERALL_oldyoung")
    plt.show()

def plot_multi_health(G, health):
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
    plt.ylabel('Network Exposure')
    ax = plt.gca()
    #ax.set_ylim([0, 1])
    plt.savefig(f"health/redAverage_500_MUL_oldyoung")
    plt.show()

def plot_health_variance(G, health):
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
    plt.ylabel('Variance of Network Exposure')

    plt.savefig(f"health/redAverage_500_VAR_oldyoung")
    plt.show()