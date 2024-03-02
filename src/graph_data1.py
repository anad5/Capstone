import networkx as nx
import os

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
