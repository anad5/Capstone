import urns
import numpy as np
import util
import networkx as nx



INITIAL_BLUE = 2
INITIAL_RED = 2
DELTA_BLUE = 1
DELTA_RED = 1
connections = np.matrix([[0, 1], [1,0]])

node_1 = urns.Urn(blue_b=INITIAL_BLUE, red_b=INITIAL_BLUE, delta_red=DELTA_RED, delta_blue=DELTA_BLUE)
node_2 = urns.Urn(blue_b=INITIAL_BLUE, red_b=INITIAL_BLUE, delta_red=DELTA_RED, delta_blue=DELTA_BLUE)

graph = nx.random_geometric_graph(200, 0.125)

edge_trace = util.generate_edge_trace(graph)
node_trace = util.generate_node_trace(graph)

util.generate_figure(edge_trace, node_trace)