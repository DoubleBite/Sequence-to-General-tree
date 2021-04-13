"""This module provides the utility function `create_tree()` for expression visualization.

Specifically, `create_tree()` does the following:
    1. Parses the input expression using `ast` module
    2. Transoforms the graph information into `dot` statements
Then the statements can be visualized by `Graphviz` python api

    Usage example
    -----------------------------------
    >>> from graphviz import Source
    >>> from visualization import create_tree
    >>> dot_string = create_tree("2+5")
    >>> Source(dot_string) # This generates the tree in Jupyter output panel

References:
    [1] sympy dot visualization: https://github.com/sympy/sympy/blob/master/sympy/printing/dot.py
    [2] Graphviz style table: https://graphviz.org/doc/info/attrs.html
    [3] dot language introduction: https://en.wikipedia.org/wiki/DOT_(graph_description_language)

TODO:
    1. Add typing to arguments
    2. Doctest
    3. Seperate the traversal and node to dot process
"""

import ast


######################################################
# TEMPLATES
######################################################

GRAPH_TEMPLATE = \
    """
    digraph{{

        # Graph style
        "ordering"="out"
        "rankdir"="TD"
        "nodesep"=0.6

        #########
        # Nodes #
        #########
        {nodes}

        #########
        # Edges #
        #########
        {edges}

    }}"""

NODE_TEMPLATE = '"{node_name}" [{node_attrs}];'

EDGE_TEMPLATE = '"{node_start}" -> "{node_end}" ["arrowsize"=0.6];'


######################################################
# STYLES
######################################################
BASIC_STYLE = {
    "shape": "circle",
    "width": 0.6,
    "fixedsize": "true",
}

NUMBER_STYLE = {
    **BASIC_STYLE,
    "color": "black",
}

VARIABLE_STYLE = {
    **BASIC_STYLE,
    "color": "blue",
}

OPERATOR_STYLE = {
    **BASIC_STYLE,
    "color": "royalblue", "style": "filled", "fontcolor": "white",
}

FORMULA_STYLE = {
    **BASIC_STYLE,
    "color": "firebrick1", "style": "filled", "fontcolor": "white",
}

STYLE_MAP = {
    "num": NUMBER_STYLE,
    "var": VARIABLE_STYLE,
    "op": OPERATOR_STYLE,
    "formula": FORMULA_STYLE
}


######################################################
# FUNCTIONS
######################################################


def create_dot_tree(expr):
    """Create the dot source code of the expression tree for the input expression

    See the module docstring for usage example
    """
    root = ast.parse(expr, mode="eval")
    all_nodes = []
    all_edges = []

    for node in ast.walk(root):
        if isinstance(node, ast.BinOp) \
                or isinstance(node, ast.Call):
            head_node = node  # Convert this with walrus operator after python 3.8

            # Get the dot code for the head node and its direct children
            local_nodes = local_dotnodes(head_node)
            local_edges = local_dotedges(head_node)
            all_nodes.extend([node for node in local_nodes
                              if node not in all_nodes])
            all_edges.extend(local_edges)

    # Postprocess nodes and edges
    all_nodes = "\n\t".join(all_nodes)
    all_edges = "\n\t".join(all_edges)

    return GRAPH_TEMPLATE.format(nodes=all_nodes, edges=all_edges)


def local_dotnodes(head_node):
    """Generate the dot code for the input node and its direct children
    """
    nodes = []
    if isinstance(head_node, ast.BinOp):
        head_name, head_type = get_node_info(head_node)
        left_name, left_type = get_node_info(head_node.left)
        right_name, right_type = get_node_info(head_node.right)
        nodes.append(generate_dotnode(head_name, head_type))
        nodes.append(generate_dotnode(left_name, left_type))
        nodes.append(generate_dotnode(right_name, right_type))
    elif isinstance(head_node, ast.Call):
        head_name, head_type = get_node_info(head_node)
        nodes.append(generate_dotnode(head_name, head_type))

        children = [get_node_info(arg)
                    for arg in head_node.args]
        nodes.extend([generate_dotnode(*child)
                      for child in children])
    return nodes


def local_dotedges(head_node):
    """Generate the dot code for the edges between the input node and its direct children
    """
    edges = []
    if isinstance(head_node, ast.BinOp):
        head_name, _ = get_node_info(head_node)
        left_name, _ = get_node_info(head_node.left)
        right_name, _ = get_node_info(head_node.right)
        edge_left = EDGE_TEMPLATE.format(
            node_start=head_name, node_end=left_name)
        edge_right = EDGE_TEMPLATE.format(
            node_start=head_name, node_end=right_name)
        edges.append(edge_left)
        edges.append(edge_right)
    elif isinstance(head_node, ast.Call):
        head_name, _ = get_node_info(head_node)
        child_names = [get_node_info(arg)[0]
                       for arg in head_node.args]
        for cname in child_names:
            edge = EDGE_TEMPLATE.format(
                node_start=head_name, node_end=cname)
            edges.append(edge)
    return edges


def get_node_info(ast_node):
    """Get the node name and node type.

    Node name includes the position offset as postfix.
    For example, "Add" would become "Add_2", so that we can distinguish different addition nodes.
    """
    # Get node name and node type
    if isinstance(ast_node, ast.BinOp):
        if isinstance(ast_node.op, ast.Add):
            node_name = "Add"
        elif isinstance(ast_node.op, ast.Sub):
            node_name = "Sub"
        elif isinstance(ast_node.op, ast.Mult):
            node_name = "Mult"
        elif isinstance(ast_node.op, ast.Div):
            node_name = "Div"
        elif isinstance(ast_node.op, ast.Pow) or isinstance(ast_node.op, ast.BitXor):
            node_name = "Pow"
        node_type = "op"
    elif isinstance(ast_node, ast.Num):
        node_name = str(ast_node.n)
        node_type = "num"
    elif isinstance(ast_node, ast.Name):
        node_name = str(ast_node.id)
        node_type = "var"
    elif isinstance(ast_node, ast.Call):
        node_name = ast_node.func.id
        node_type = "formula"

    node_name = f"{node_name}_{ast_node.col_offset}"
    return node_name, node_type


def generate_dotnode(node_name, node_type):
    """Generate the dot code for a given node

    It processes the `node_name` so that it becomes a proper label to show on the graph node,
    and also chooses the corresponding style based on the `node_type`
    Finally it generates the dot statement for this node.

        Usage example:
        ----------------------------------------------
        # the label to show on the node will be "+"
        >>> dot_node = generate_dotnode("Add_1", "op")
        >>> print(dot_node)
        '"Add_1" ["color"="royalblue", "fixedsize"="true", "fontcolor"="white", \
            "label"="+", "shape"="circle", "style"="filled", "width"="0.6"];'
    """

    OP_MAP = {
        "Add": "+",
        "Sub": "-",
        "Mult": "*",
        "Div": "/",
        "Pow": "^"
    }

    # Generate the label
    if node_type == "op":
        name = node_name.rsplit("_", maxsplit=1)[0]
        label = OP_MAP[name]
    elif node_type == "num":
        label = node_name.rsplit("_", maxsplit=1)[0]
    elif node_type == "var":
        label = node_name.rsplit("_", maxsplit=1)[0]
    elif node_type == "formula":
        label = node_name.rsplit("_", maxsplit=1)[0]

    # Because graphviz won't do linebreak, we need to do it manually
    # E.g. circle_area --> circle\\narea
    if "_" in label:
        label = "\\n".join(label.split("_", maxsplit=1))

    # Set style
    style = STYLE_MAP[node_type]
    style["label"] = label  # Add node label to styles
    style["fontsize"] = select_font_size(label)
    node_attrs = attrprint(style)

    return NODE_TEMPLATE.format(node_name=node_name, node_attrs=node_attrs)


###################################################################
# UTILITY FUNCTIONS
###################################################################


def attrprint(d, delimiter=', '):
    """ Print a dictionary of attributes

    Adapted from: https://github.com/sympy/sympy/blob/master/sympy/printing/dot.py

        Examples
        ========
        >>> from sympy.printing.dot import attrprint
        >>> print(attrprint({'color': 'blue', 'shape': 'ellipse'}))
        "color"="blue", "shape"="ellipse"
    """
    return delimiter.join('"%s"="%s"' % item for item in sorted(d.items()))


def select_font_size(string):
    """Select proper font size for graphviz label.
    """

    if "\\n" in string:
        string = max(string.split("\\n"), key=len)

    if len(string) <= 2:
        return 20.0
    elif 2 < len(string) and len(string) <= 4:
        return 15.0
    elif 4 < len(string) and len(string) <= 6:
        return 12.0
    elif 6 < len(string) and len(string) <= 10:
        return 10.0
    else:
        return 7.0
