# IMPLEMENTING FUNCTIONS FOR HEURISTIC SOLUTIONS 

import networkx as nx

def calc_susceptibility(graph, node, red_balls_attribute, total_balls_attribute):
    """
    Calculate the susceptibility of a node based on the proportion of red balls in its local urn and neighboring urns.

    Args:
        graph: Networkx graph object.
        node: Node for which susceptibility is to be calculated.
        red_balls_attribute: Attribute representing the number of red balls in each urn.
        total_balls_attribute: Attribute representing the total number of balls in each urn.

    Returns:
        susceptibility_num: Susceptibility score for the node.
    """
    # Get neighboring nodes
    neighbors = graph.neighbors(node)
    
    # Initialize variables to store total red balls and total balls
    total_red_balls = graph.nodes[node][red_balls_attribute]
    total_balls = graph.nodes[node][total_balls_attribute]

    # Sum up the red and total balls from the neighbors' urns
    for neighbor in neighbors:
        total_red_balls += graph.nodes[neighbor][red_balls_attribute]
        total_balls += graph.nodes[neighbor][total_balls_attribute]

    # Calculate the proportion of red balls in the super urn
    if total_balls > 0:
        susceptibility_num = total_red_balls / total_balls 
    else: 
        susceptibility_num = 0
    return susceptibility_num

def calc_degree(graph, node):
    """
    Calculate the degree of a node in the graph.

    Args:
        graph: Networkx graph object.
        node: Node for which degree is to be calculated.

    Returns:
        deg: Degree of the node.
    """
    if node in graph:
        deg = graph.degree(node)
    else:    
        deg = 0
    return deg  

def calc_centrality(graph, node):
    """
    Calculate the centrality of a node based on the inverse of the sum of shortest path lengths from the node to all other nodes.

    Args:
        graph: Networkx graph object.
        node: Node for which centrality is to be calculated.

    Returns:
        centrality: Centrality score for the node.
    """
    # Calculate shortest path lengths from the node to all other nodes
    node_path = nx.single_source_shortest_path_length(graph, node)
    
    # Calculate the sum of shortest path lengths
    distance = sum(node_path.values())
    
    # Calculate centrality based on the inverse of the sum of shortest path lengths
    if distance > 0:
        centrality = 1.0 / distance 
    else:
        centrality = 0
    return centrality 

def score_midpoint(scores):
    """
    Calculate the midpoint score from a dictionary of scores.

    Args:
        scores: Dictionary containing scores for nodes.

    Returns:
        nodes_above_midpoint: List of nodes with scores above the midpoint.
    """
    # Calculate the average score
    average_score = sum(scores.values()) / len(scores)
    
    # Identify nodes with scores above the average
    nodes_above_midpoint = [node for node, score in scores.items() if score > average_score]
    
    return nodes_above_midpoint