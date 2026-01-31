import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import plotly.graph_objects as go

# =====================================================
# NODE COLORS (paper style)
# =====================================================

COLORS = {
    "Finding": "#ff6b6b",
    "Location": "#6bcBef",
    "Pattern": "#ffd93d",
    "Disease": "#6dd56d",
    "Treatment": "#d291ff",
    "Guideline": "#ffa600"
}


# =====================================================
# BUILD GRAPH (dynamic)
# =====================================================

def build_graph(cnn_outputs):

    G = nx.DiGraph()

    # ---------- Add CNN findings dynamically ----------
    for finding, conf, location in cnn_outputs:

        G.add_node(finding, type="Finding")
        G.add_node(location, type="Location")

        G.add_edge(finding, location, label="located_in")

    # ---------- Medical knowledge base ----------
    rules = [

        ("PatchyOpacity", "BacterialPattern", "suggests"),
        ("GroundGlass", "ViralPattern", "suggests"),

        ("BacterialPattern", "Pneumonia", "indicates"),
        ("ViralPattern", "Normal", "possible"),

        ("Pneumonia", "Antibiotics", "treated_with"),
        ("Normal", "Rest", "managed_with"),

        ("Pneumonia", "WHO_Guideline", "supported_by")
    ]

    for s, t, l in rules:
        G.add_edge(s, t, label=l)

    # ---------- Add missing nodes with types ----------
    types = {
        "BacterialPattern": "Pattern",
        "ViralPattern": "Pattern",
        "Pneumonia": "Disease",
        "Normal": "Disease",
        "Antibiotics": "Treatment",
        "Rest": "Treatment",
        "WHO_Guideline": "Guideline"
    }

    for n, t in types.items():
        if n not in G:
            G.add_node(n, type=t)
        else:
            G.nodes[n]["type"] = t

    return G


# =====================================================
# REASONING (auto path detection)
# =====================================================

def find_reasoning_path(G):

    # BFS to disease
    for node, data in G.nodes(data=True):
        if data["type"] == "Finding":
            for target in nx.descendants(G, node):
                if G.nodes[target]["type"] == "Disease":
                    return nx.shortest_path(G, node, target)

    return []


# =====================================================
# STATIC CLEAN FIGURE
# =====================================================

def save_static_graph(G, path):

    pos = nx.spring_layout(G, k=1.4, seed=42)

    plt.figure(figsize=(14,10))

    # draw nodes
    for t, color in COLORS.items():
        nodes = [n for n,d in G.nodes(data=True) if d["type"] == t]
        nx.draw_networkx_nodes(G, pos, nodelist=nodes, node_color=color, node_size=2000)

    nx.draw_networkx_labels(G, pos, font_size=9)

    # edges
    nx.draw_networkx_edges(G, pos, width=2, arrows=True)

    labels = nx.get_edge_attributes(G, "label")
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, font_size=8)

    legend = [Line2D([0],[0], marker='o', color='w', label=k,
                     markerfacecolor=v, markersize=12) for k,v in COLORS.items()]
    plt.legend(handles=legend)

    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


# =====================================================
# INTERACTIVE PLOTLY GRAPH
# =====================================================

def save_interactive_graph(G, path, highlight_path=None):

    pos = nx.spring_layout(G, k=1.4, seed=42)

    edge_x, edge_y = [], []

    for e in G.edges():
        x0,y0 = pos[e[0]]
        x1,y1 = pos[e[1]]
        edge_x += [x0,x1,None]
        edge_y += [y0,y1,None]

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1,color='#888'),
        hoverinfo='none',
        mode='lines'
    )

    node_x, node_y, colors = [],[],[]

    for node,data in G.nodes(data=True):
        x,y = pos[node]
        node_x.append(x)
        node_y.append(y)
        colors.append(COLORS[data["type"]])

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=list(G.nodes()),
        textposition="top center",
        marker=dict(size=25, color=colors)
    )

    fig = go.Figure(data=[edge_trace,node_trace])
    fig.write_html(path)


# =====================================================
# NEO4J EXPORT
# =====================================================

def export_neo4j(G, path):

    with open(path, "w") as f:

        for n,data in G.nodes(data=True):
            f.write(f"CREATE (:{data['type']} {{name:'{n}'}});\n")

        for s,t,d in G.edges(data=True):
            f.write(
                f"MATCH (a {{name:'{s}'}}),(b {{name:'{t}'}}) "
                f"CREATE (a)-[:{d['label']}]->(b);\n"
            )


# =====================================================
# RDF / OWL EXPORT
# =====================================================

def export_rdf(G, path):

    with open(path,"w") as f:
        for s,t,d in G.edges(data=True):
            f.write(f":{s} :{d['label']} :{t} .\n")


# =====================================================
# MAIN
# =====================================================

if __name__ == "__main__":

    cnn_outputs = [
        ("PatchyOpacity",0.82,"LowerLobe"),
        ("GroundGlass",0.31,"LowerLobe")
    ]

    G = build_graph(cnn_outputs)

    save_static_graph(G,"results/graph.png")

    save_interactive_graph(G,"results/graph.html")

    export_neo4j(G,"results/graph.cypher")

    export_rdf(G,"results/graph.ttl")

    print("Graphs exported successfully.")
