import networkx as nx
import matplotlib.pyplot as plt
import os


# =====================================================
# COLOR SCHEME (IEEE publication style)
# =====================================================

COLORS = {

    "Input": "#94a3b8",
    "CNN": "#3b82f6",
    "Sign": "#ef4444",
    "Reasoning": "#10b981",
    "KG": "#0ea5e9",
    "Disease": "#22c55e",
    "Treatment": "#a855f7",
    "Guideline": "#f59e0b",
    "Output": "#dc2626"
}


# =====================================================
# BUILD GRAPH
# =====================================================

def build_graph(sign_probs):

    G = nx.DiGraph()

    # Layer 0 — Input
    G.add_node("Chest X-ray Image", type="Input")

    # Layer 1 — CNN
    G.add_node("EfficientNetV2 CNN\n(Feature Extraction)", type="CNN")

    G.add_edge(
        "Chest X-ray Image",
        "EfficientNetV2 CNN\n(Feature Extraction)",
        label="input"
    )


    # Layer 2 — Sign prediction
    sign_nodes = []

    for sign, prob in sign_probs.items():

        node = f"{sign}\nP={prob:.2f}"

        G.add_node(node, type="Sign")

        G.add_edge(
            "EfficientNetV2 CNN\n(Feature Extraction)",
            node,
            label="predicts"
        )

        sign_nodes.append(node)


    # Layer 3 — Symbolic reasoning
    G.add_node("Symbolic Reasoning Layer", type="Reasoning")

    for node in sign_nodes:

        G.add_edge(
            node,
            "Symbolic Reasoning Layer",
            label="input"
        )


    # Layer 4 — Knowledge graph
    G.add_node("Medical Knowledge Graph", type="KG")

    G.add_edge(
        "Symbolic Reasoning Layer",
        "Medical Knowledge Graph",
        label="semantic inference"
    )


    # Layer 5 — Disease
    G.add_node("Pneumonia", type="Disease")

    G.add_edge(
        "Medical Knowledge Graph",
        "Pneumonia",
        label="diagnosis"
    )


    # Layer 6 — Treatment
    G.add_node("Antibiotics", type="Treatment")

    G.add_edge(
        "Pneumonia",
        "Antibiotics",
        label="treated_with"
    )


    # Layer 7 — Guideline
    G.add_node("WHO Clinical Guideline", type="Guideline")

    G.add_edge(
        "Pneumonia",
        "WHO Clinical Guideline",
        label="supported_by"
    )


    # Layer 8 — Final explainable output
    G.add_node(
        "Explainable Clinical Decision:\nPneumonia Detected",
        type="Output"
    )

    G.add_edge(
        "Pneumonia",
        "Explainable Clinical Decision:\nPneumonia Detected",
        label="final_output"
    )

    return G


# =====================================================
# DRAW LAYERED GRAPH (NO OVERLAP)
# =====================================================

def draw_graph(G):

    os.makedirs("results", exist_ok=True)

    plt.figure(figsize=(16,14))

    pos = {}

    # Fixed hierarchical layout

    pos["Chest X-ray Image"] = (0, 10)

    pos["EfficientNetV2 CNN\n(Feature Extraction)"] = (0, 8)

    sign_positions = [-4, -1.5, 1.5, 4]

    sign_nodes = [n for n,d in G.nodes(data=True) if d["type"]=="Sign"]

    for i,node in enumerate(sign_nodes):

        pos[node] = (sign_positions[i], 6)


    pos["Symbolic Reasoning Layer"] = (0, 4)

    pos["Medical Knowledge Graph"] = (0, 2)

    pos["Pneumonia"] = (0, 0)

    pos["Antibiotics"] = (-3, -2)

    pos["WHO Clinical Guideline"] = (3, -2)

    pos["Explainable Clinical Decision:\nPneumonia Detected"] = (0, -4)


    # Draw nodes by type

    for t,color in COLORS.items():

        nodes = [n for n,d in G.nodes(data=True) if d["type"]==t]

        nx.draw_networkx_nodes(

            G,
            pos,
            nodelist=nodes,
            node_color=color,
            node_size=3500,
            edgecolors="black"
        )


    nx.draw_networkx_labels(

        G,
        pos,
        font_size=9,
        font_weight="bold"
    )


    nx.draw_networkx_edges(

        G,
        pos,
        arrows=True,
        width=2
    )


    edge_labels = nx.get_edge_attributes(G,'label')

    nx.draw_networkx_edge_labels(

        G,
        pos,
        edge_labels=edge_labels,
        font_size=8
    )


    # Layer labels (important for paper)

    plt.text(-7,10,"Layer 0: Medical Image Input",fontsize=11,fontweight="bold")
    plt.text(-7,8,"Layer 1: Neural Feature Extraction",fontsize=11,fontweight="bold")
    plt.text(-7,6,"Layer 2: Medical Sign Prediction",fontsize=11,fontweight="bold")
    plt.text(-7,4,"Layer 3: Symbolic Reasoning",fontsize=11,fontweight="bold")
    plt.text(-7,2,"Layer 4: Knowledge Graph Inference",fontsize=11,fontweight="bold")
    plt.text(-7,0,"Layer 5: Disease Diagnosis",fontsize=11,fontweight="bold")
    plt.text(-7,-2,"Layer 6: Clinical Knowledge",fontsize=11,fontweight="bold")
    plt.text(-7,-4,"Layer 7: Explainable Output",fontsize=11,fontweight="bold")


    plt.title(

        "Neuro-Symbolic Knowledge Graph for Explainable Medical Diagnosis",

        fontsize=16,
        fontweight="bold"
    )


    plt.axis("off")


    plt.savefig(

        "results/knowledge_graph_PUBLICATION_READY.png",

        dpi=300,
        bbox_inches="tight"
    )


    plt.close()


# =====================================================
# MAIN
# =====================================================

if __name__ == "__main__":

    sign_predictions = {

        "Opacity": 0.92,
        "Consolidation": 0.81,
        "Infiltration": 0.45,
        "Inflammation": 0.63
    }


    G = build_graph(sign_predictions)

    draw_graph(G)

    print("\nFINAL publication-ready knowledge graph generated.")
    print("Saved at: results/knowledge_graph_PUBLICATION_READY.png")
