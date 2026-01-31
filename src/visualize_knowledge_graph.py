import networkx as nx
import matplotlib.pyplot as plt

# ---------------------------------
# Build directed graph
# ---------------------------------
G = nx.DiGraph()

# Node groups
nodes = {
    "Finding": ["PatchyOpacity", "GroundGlass", "HighActivationLowerLobe"],
    "Location": ["LowerLobe"],
    "Pattern": ["BacterialPattern", "ViralPattern"],
    "Disease": ["Pneumonia", "Normal"],
    "Treatment": ["Antibiotics", "Rest"],
    "Guideline": ["WHO_Guideline"]
}

# Add nodes
for t, items in nodes.items():
    for n in items:
        G.add_node(n, type=t)

# ---------------------------------
# Edges with directions
# ---------------------------------
edges = [
    ("HighActivationLowerLobe", "LowerLobe", "located_in"),
    ("PatchyOpacity", "BacterialPattern", "suggests"),
    ("GroundGlass", "ViralPattern", "suggests"),
    ("BacterialPattern", "Pneumonia", "indicates"),
    ("ViralPattern", "Normal", "possible"),
    ("Pneumonia", "Antibiotics", "treated_with"),
    ("Normal", "Rest", "managed_with"),
    ("Pneumonia", "WHO_Guideline", "supported_by")
]

for s, t, l in edges:
    G.add_edge(s, t, label=l)

# ---------------------------------
# Hierarchical layout (flow left → right)
# ---------------------------------
layer_map = {
    "Finding": 0,
    "Location": 1,
    "Pattern": 2,
    "Disease": 3,
    "Treatment": 4,
    "Guideline": 5
}

pos = {}
for node in G.nodes:
    x = layer_map[G.nodes[node]["type"]]
    y = hash(node) % 10  # vertical spacing
    pos[node] = (x, y)


# ---------------------------------
# Colors
# ---------------------------------
colors = {
    "Finding": "#ff9999",
    "Location": "#8ecae6",
    "Pattern": "#ffd166",
    "Disease": "#80ed99",
    "Treatment": "#c77dff",
    "Guideline": "#ffb703"
}

node_colors = [colors[G.nodes[n]["type"]] for n in G.nodes]


# ---------------------------------
# DRAW (important arrows settings)
# ---------------------------------
plt.figure(figsize=(14, 8))

nx.draw_networkx_nodes(
    G, pos,
    node_color=node_colors,
    node_size=3000
)

nx.draw_networkx_labels(
    G, pos,
    font_size=10,
    font_weight="bold"
)

# 🔴 THIS IS THE KEY PART
nx.draw_networkx_edges(
    G, pos,
    arrows=True,
    arrowstyle='-|>',   # clean arrow
    arrowsize=30,       # BIG arrows
    width=2.5,
    connectionstyle='arc3,rad=0.1'
)

edge_labels = nx.get_edge_attributes(G, 'label')

nx.draw_networkx_edge_labels(
    G, pos,
    edge_labels=edge_labels,
    font_size=9
)

plt.axis("off")
plt.tight_layout()
plt.savefig("results/knowledge_graph_final.png", dpi=300)
plt.show()
