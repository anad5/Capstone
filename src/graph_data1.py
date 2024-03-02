import networkx as nx
import os

def combine_graphs(circle_files_directory, combined_file=None):
    # Initialize an empty graph
    G = nx.Graph()
    
    # Add edges from the combined graph file if provided
    if combined_file:
        with open(combined_file, 'r') as f:
            for line in f:
                try:
                    parts = line.strip().split()
                    if len(parts) == 2:
                        src, dst = map(int, parts)
                        G.add_edge(src, dst)
                except ValueError:
                    # Handle the exception, e.g., print an error message or pass
                    print(f"Skipping line with invalid integers: {line.strip()}")

    # Add edges from each of the edge files in the directory
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
                    # Handle the exception, e.g., print an error message or pass
                    print(f"Skipping line with invalid integers: {line.strip()}")

    return G
