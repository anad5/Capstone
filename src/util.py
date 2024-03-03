import plotly.graph_objects as go
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import random as rd

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

def generate_figure(edge_trace, node_trace, health, graph, iterations):
    fig = go.Figure(data=[edge_trace, node_trace],
             layout=go.Layout(
                frames=go.Frame(data=node_trace),
                title='<br>Network graph made with Python',
                titlefont_size=16,
                #width=800,
                #height=800, 
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                annotations=[ dict(
                    text="Python code: <a href='https://plotly.com/ipython-notebooks/network-graphs/'> https://plotly.com/ipython-notebooks/network-graphs/</a>",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.005, y=-0.002 ) ],
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)),
                )
    fig['layout']['sliders']={
        'active': 0,
        'yanchor': 'top',
        'xanchor': 'left',
        'currentvalue': {
            'font': {'size': 20},
            'prefix': 'Iterations',
            'visible': True,
            'xanchor': 'right'
        },
        'transition': {'duration': 300, 'easing': 'cubic-in-out'},
        'pad': {'b': 10, 't': 50},
        'len': 0.9,
        'x': 0.1,
        'y': 0,
        'steps': [...]
    }
    fig['layout']['updatemenus'] = [
    {
        'buttons': [
            {
                'args': [None, {'frame': {'duration': 500, 'redraw': False},
                         'fromcurrent': True, 'transition': {'duration': 300, 'easing': 'quadratic-in-out'}}],
                'label': 'Play',
                'method': 'animate'
            },
            {
                'args': [[None], {'frame': {'duration': 0, 'redraw': False}, 'mode': 'immediate',
                'transition': {'duration': 0}}],
                'label': 'Pause',
                'method': 'animate'
            }
        ],
        'direction': 'left',
        'pad': {'r': 10, 't': 87},
        'showactive': False,
        'type': 'buttons',
        'x': 0.1,
        'xanchor': 'right',
        'y': 0,
        'yanchor': 'top'
    }
    ]
    for iteration in iterations:
        data_dict = {}

    fig.show()

def generate_graph(adj_matrix_path, skiprows=0):
    np_array = np.loadtxt(open(adj_matrix_path, "rb"), delimiter=",", skiprows=skiprows)
    graph = nx.from_numpy_array(np_array)
    
    return graph


#New
def animate_nodes(G, node_colors, scalarmappaple, colormap, pos=None, *args, **kwargs):

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

def update_super(graph, num_nodes):
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

def pull_ball(graph, delta_blue, delta_red, num_nodes):
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

def init_urns(graph, num_nodes):
    for node in range(num_nodes):
        # i, red=2, blue=2, total=4, super_red=2, super_blue=2, super_total=4, health=[1], pos=(i,1)
        graph.nodes[node]['red'] = 2
        graph.nodes[node]['blue'] = 2
        graph.nodes[node]['total'] = 4
        graph.nodes[node]['super_red'] = 2
        graph.nodes[node]['super_blue'] = 2
        graph.nodes[node]['super_total'] = 2
        graph.nodes[node]['health'] = [0.5]


