import pandas as pd
import networkx as nx
from networkx.algorithms import community

def insert_csv_to_graph(csv_file):
    # Read the CSV file
    df = pd.read_csv(csv_file).fillna(-1)
    
    # Create an empty directed graph
    graph = nx.DiGraph()
    
    # Iterate over each row in the CSV and add edges to the graph
    for _, row in df.iterrows():
        if row['P-Value pcc_wo_organelle_pixels vs pcc_random_pixels']<0.05 and row['P-Value pcc_wo_organelle_pixels vs pcc_random_pixels']>0.0:
            source = row['source']
            target = row['target']
            weight = 1 / row['pcc_wo_organelle_pixels']
            
            if weight > 1.01:
                
                # Add the edge to the graph
                graph.add_edge(source, target, weight=weight, **row.to_dict())
    
    partition = nx.community.greedy_modularity_communities(graph,weight='weight',resolution=0.9)
    node_community_dict = {}

    # Loop through each community in the partition to enumerate the communities
    for i, communities in enumerate(partition):
        for node in communities:
            node_community_dict[node] = i   
            
    nx.set_node_attributes(graph, node_community_dict, 'community') 
        
    # Add location_in_cell metadata to all nodes
    for node in graph.nodes:
        location_in_cell = df.loc[df['source'] == node, 'location_in_cell'].values[0]
        graph.nodes[node]['location_in_cell'] = location_in_cell
    
    return graph

# Path to the CSV file
csv_file = '/sise/home/lionb/predictions_correlations_constant_main_organelle_precent_pixels_0.001_in_image.csv'

# Insert the CSV data into a NetworkX graph
graph = insert_csv_to_graph(csv_file)

# Save the graph as a GEXF file
output_file = '/sise/home/lionb/correlations_graph.gexf'
nx.write_gexf(graph, output_file)