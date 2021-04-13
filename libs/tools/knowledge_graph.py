import itertools

import networkx as nx
import matplotlib.pyplot as plt
import json


formula_to_args = {
    "square_perimeter": ["square_side"],
    "square_area": ["square_side"],
    "cubic_volume": ["square_side"],
    "circle_area": ["circle_area"],
    "circumference_radius": ["circle_area"],
    "circumference_diameter": ["circle_area"],
    "triangle_area": ["triangle_base", "triangle_height"],
    "rectangle_area": ["rectangle_length", "rectangle_width"],
    "rectangle_perimeter": ["rectangle_length", "rectangle_width"],
    "cuboid_volume": ["rectangle_length", "rectangle_width", "rectangle_height"],
    "cuboid_surface": ["rectangle_length", "rectangle_width", "rectangle_height"],
    "+": ["addition_arg1", "addition_arg2"],
    "-": ["subtraction_arg1", "subtraction_arg2"],
    "*": ["multiplication_arg1", "multiplication_arg2"],
    "/": ["division_arg1", "division_arg2"],
    "^": ["power_arg1", "power_arg2"],
}


def create_KGs_with_id_table(op_kg, geom_kg):
    node_name_to_id = {}
    G1, node_count, node_name_to_id1 = create_math_KG(op_kg)
    G2, _, node_name_to_id2 = create_math_KG(geom_kg, node_count)
    G = nx.compose(G1, G2)

    node_name_to_id.update(node_name_to_id1)
    node_name_to_id.update(node_name_to_id2)

    nodes = list(G.nodes())
    edges = list(G.edges())

    return G, nodes, edges, node_name_to_id


def create_math_KG(sub_graphs, node_count=0):

    G = nx.Graph()
    node_name_to_id = {}

    for sub_graph in sub_graphs:
        graph_name = sub_graph["subgraph_name"]
        nodes = sub_graph["nodes"]
        nodes = [(node[0]+node_count, node[1]) for node in nodes]
        G.add_nodes_from(nodes)

        for node in nodes:
            node_id = node[0]
            if node[1]["type"] in ["argument", "quantity"]:
                node_name = f"{graph_name}_{node[1]['name']}"
            else:
                node_name = node[1]['name']
            node_name_to_id[node_name] = node_id

        if "edges" in sub_graph:
            edges = sub_graph["edges"]
            edges = [(edge[0]+node_count, edge[1]+node_count)
                     for edge in edges]
            G.add_edges_from(edges)

        node_count += len(nodes)

    # Add links between ops and root
    root_and_op_nodes = [n for (n, n_type) in
                         nx.get_node_attributes(G, 'type').items() if n_type in ["root", "operator", "object"]]
    G.add_edges_from(itertools.combinations(root_and_op_nodes, 2))

    return G, node_count, node_name_to_id


def visualize_math_KG(knowledge_graph):
    G = knowledge_graph
    plt.figure(figsize=(20, 10))
    pos = nx.spring_layout(G)
    names = nx.get_node_attributes(G, 'name')
    color_map = {
        "root": "gold",
        "operator": "red",
        "object": "red",
        "argument": "blue",
        "quantity": "blue",
    }
    colors = [color_map[G.nodes[node]["type"]]for node in G.nodes()]

    nx.draw_networkx(G, labels=names, node_size=2000,
                     font_size=15, font_color="white", node_color=colors)
