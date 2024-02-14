import random
import networkx as nx
from pyvis.network import Network
from IPython.display import display, IFrame
import ipywidgets as widgets

# Function to run consensus Polya urn model and visualize the network
def interactive_consensus_polya_urn(G, iterations):
    urn = list(G.nodes())  # Initialize urn with node labels
    results = {node: 0 for node in G.nodes()}  # Initialize results dict with counts

    for _ in range(iterations):
        chosen_node = random.choice(urn)  # Randomly choose a node
        results[chosen_node] += 1  # Update count for chosen node
        
        urn.append(chosen_node)  # Always add chosen node back to urn

    # Adjusting the return format to match the original
    results = {node: [results[node]] for node in results}

    # Visualize the updated network
    net = Network(notebook=True, height='750px', width='100%', bgcolor="#222222", font_color="white")
    net.from_nx(G)
    # You can customize the nodes here based on the results if needed
    net.show("interactive_facebook_network.html")
    display(IFrame('interactive_facebook_network.html', width='100%', height='750px'))

# Load your graph (Make sure to replace this with the correct path or method to load your graph)
edges_file_path = "/Users/madeline/Downloads/archive (1)/facebook/facebook/414.edges"  # Update this path
G = nx.read_edgelist(edges_file_path)

# Widget for selecting the number of iterations
iterations_slider = widgets.IntSlider(value=1000, min=100, max=5000, step=100, description='Iterations:')

# Button to run the simulation
run_button = widgets.Button(description='Run Simulation')

# Output widget to display the interactive network visualization
output = widgets.Output()

def on_run_button_clicked(b):
    with output:
        output.clear_output()
        interactive_consensus_polya_urn(G, iterations_slider.value)

run_button.on_click(on_run_button_clicked)

# Display the widgets
display(iterations_slider, run_button, output)


