import logging

import networkx as nx

logger = logging.getLogger(__name__)


def concatenate_edge_types(G, node1, node2):
    edge_types = []
    edges = {}

    for key, data in G.get_edge_data(node1, node2).items():
        if "type" in data and data["type"]:
            edge_type = data["type"]
            dataflow = data["label"] if "label" in data else None

            if edge_type not in edges:
                edges[edge_type] = {dataflow}
            else:
                edges[edge_type].add(dataflow)

    for edge_type, dataflows in edges.items():
        if edge_type in ["DDG", "REF"]:
            flow_str_list = ",".join(filter(None, dataflows))
            edge_types.append(f"<{edge_type}>{flow_str_list}</{edge_type}>")
        else:
            edge_types.append(f"<{edge_type}>")

    return f"{' '.join(edge_types)}".strip()


def structured_vdg_traversal(graph, node_id=None, visited=None):
    if not graph:
        return "()"

    if node_id is None:
        components = list(nx.connected_components(graph.to_undirected()))

        if not components:
            logger.debug("No components found in the graph.")
            return "()"

        component_roots = [min(comp) for comp in components]
        assert component_roots, f"Cannot process graph having empty components!"

        if len(components) > 1:
            logger.debug(f"Generating VDG for unconnected graphs: {graph}")
            return "<vul-sep>".join(
                [structured_vdg_traversal(graph, root) for root in component_roots]
            )
        else:
            logger.debug(f"Generating VDG for connected graph: {graph}")

        node_id = min(component_roots)

    if visited is None:
        visited = set()

    current_node = graph.nodes[node_id]
    node_value = current_node["code"].strip()

    if node_id in visited:
        # already traversed, generate reference node
        return f"<node>{node_value}</node>"

    visited.add(node_id)

    if graph.out_degree(node_id) == 0:
        # reached leaf node
        return f"<node>{node_value}</node>"
    else:
        # walk tree and source code from top to bottom
        children = sorted(list(graph.successors(node_id)))

        child_sequences = [
            concatenate_edge_types(graph, node_id, child)
            + " "
            + structured_vdg_traversal(graph, child, visited=visited)
            for child in children
        ]

        children_string = " ".join(child_sequences)

        return f"(<node>{node_value}</node>, {children_string})"
