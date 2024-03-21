# IMPLEMENTING FUNCTIONS FOR HEURISTIC SOLUTIONS 

import networkx as nx

def calc_susceptibility(graph, node, red_balls_attribute, total_balls_attribute):
    neighbors = graph.neighbors(node)
    total_red_balls = graph.nodes[node][red_balls_attribute]
    total_balls = graph.nodes[node][total_balls_attribute]

    # sum up the red and total balls from the neighbors' urns
    for neighbor in neighbors:
        total_red_balls += graph.nodes[neighbor][red_balls_attribute]
        total_balls += graph.nodes[neighbor][total_balls_attribute]

    # calculate the proportion of red balls in the super urn
    if total_balls > 0:
        susceptibility_num = total_red_balls / total_balls 
    else: 
        susceptibility_num = 0
    return susceptibility_num

def calc_degree(graph, node):
    if node in graph:
        deg = graph.degree(node)
    else:    
        deg = 0
    return deg  

# NEED TO WORK ON THIS IT IS NOT WORKING 
def calc_centrality(graph, node):
    # original
    #centrality_dict = nx.closeness_centrality(graph)
    #return centrality_dict[node]

    # new try 
    # need to look into the different options for shortest path length
    node_path = nx.single_source_shortest_path_length(graph, node)
    distance = sum(node_path.values())
    if distance > 0:
        centrality = 1.0/distance 
    else:
        centrality = 0
    return centrality 

def score_midpoint(scores):
    # want averagge and max score 
    average_score = sum(scores.values()) / len(scores)
    #min_score = min(scores.values())
    #max_score = max(scores.values())
    #hello = min_score + max_score 
    
    # midpoint between the averagge heuristic score and max 
    #midpoint = (average_score + max_score) / 2
    #midpoint = (max_score + min_score)/2 
    #midpoint = (hello + average_score)/2
    
    # nodes above the midpoint
    nodes_above_midpoint = [node for node, score in scores.items() if score > average_score]
    
    return nodes_above_midpoint



