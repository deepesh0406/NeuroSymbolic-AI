# Neuro-Symbolic Medical Diagnosis System

Explainable AI system for Pneumonia detection using:

• CNN (EfficientNetV2)
• Grad-CAM visual evidence
• Knowledge Graph reasoning
• Symbolic rules
• Graph visualization
• Neo4j + RDF export

---

## Features

✔ Deep learning diagnosis  
✔ Visual explanation (GradCAM)  
✔ Neuro-symbolic reasoning  
✔ Directed knowledge graph  
✔ Automatic reasoning path highlight  
✔ Interactive HTML graph  
✔ Neo4j export  
✔ RDF/OWL export  

---

## Run Pipeline

### Train
python -m src.train_model

### Evaluate
python -m src.evaluate_model

### GradCAM
python -m src.gradcam_visualization

### Knowledge Graph
python -m src.knowledge_graph

---

## Outputs

results/
- roc_curve.png
- gradcam.png
- knowledge_graph_final.png
- knowledge_graph.html
- nodes.csv
- edges.csv
- knowledge_graph.owl

---

## Tech Stack

PyTorch  
NetworkX  
Plotly  
RDFlib  
GradCAM  

---

## Author
Deepesh Bhardwaj  
Final Year Project – CSE  
"# NeuroSymbolic-AI" 
