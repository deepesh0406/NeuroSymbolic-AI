# TODO: Fix knowledge_graph.py for Better Knowledge Graph Representation

## Steps to Complete:
- [x] Modify build_graph function to load nodes from nodes.csv and edges from edges.csv instead of hardcoding.
- [x] Fix export_neo4j function: change header from "relation" to "label" to match the edge data.
- [x] Fix export_rdf function: change data.get("relation", "related_to") to data.get("label", "related_to").
- [x] Update main function to save static graph as "knowledge_graph_final.png" to match existing file.
- [x] Test the changes by running the script and verifying outputs.
