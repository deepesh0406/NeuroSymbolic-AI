# Neuro-Symbolic AI for Explainable Medical Diagnosis

## 📌 Overview
This project presents a Neuro-Symbolic Artificial Intelligence framework for explainable pneumonia detection using chest X-ray images. The system combines deep learning with symbolic reasoning to provide both accurate and interpretable medical diagnosis.

Unlike traditional black-box models, this approach mimics clinical reasoning by first detecting intermediate radiological signs and then applying rule-based inference to reach a final decision.

---

## 🚀 Key Features
- EfficientNetV2-S with transfer learning  
- Multi-label medical sign prediction  
- Symbolic reasoning using knowledge graph  
- Grad-CAM based visual explainability  
- Faithfulness and localization analysis  
- ROC and Precision-Recall evaluation  
- Ablation study and model comparison  

---

## 🏗️ Architecture
The system follows a structured neuro-symbolic pipeline:

1. Input chest X-ray image  
2. Feature extraction using EfficientNetV2  
3. Prediction of intermediate medical signs  
4. Symbolic reasoning using predefined rules  
5. Final classification (Pneumonia / Normal)  
6. Explainability using Grad-CAM  

---

## 📂 Project Structure
neuro_symbolic_medical/
│
├── src/
│ ├── train_model.py
│ ├── evaluate_model.py
│ ├── dataset.py
│ ├── efficientnet_model.py
│ ├── symbolic_reasoner.py
│ ├── gradcam_visualization.py
│
├── models/
│ └── efficientnet_signs.pth
│
├── data/
│ └── chest_xray/
│
├── results/
│ ├── plots/
│ └── outputs/
│
├── plot_results.py
├── README.md

---

## ⚙️ Installation
git clone https://github.com/deepesh0406/NeuroSymbolic-AI.git

cd NeuroSymbolic-AI

python -m venv venv
venv\Scripts\activate

pip install -r requirements.txt

---

## 🏋️ Training
python src/train_model.py

---

## 📊 Evaluation
python plot_results.py

Generated outputs:
- Faithfulness Curve  
- Localization Accuracy  
- Performance Metrics  
- Model Comparison  
- Ablation Study  
- Precision-Recall Curve  

---

## 📊 Results
| Metric | Value |
|--------|------|
| Accuracy | 92.47% |
| Precision | 92.56% |
| Recall | 95.64% |
| F1 Score | 94.07% |
| AUC | 0.9666 |

---

## 🧠 Explainability
The system ensures transparent decision-making using:
- Grad-CAM for visual explanations  
- Symbolic rules for logical reasoning  
- Faithfulness evaluation for reliability  
- Localization accuracy for spatial correctness  

---

## 📚 Contribution
- Hybrid neuro-symbolic AI architecture  
- Separation of perception and reasoning  
- Clinically aligned decision process  
- Improved interpretability in medical AI  

---

## ⚠️ Limitations
- Some explainability metrics are approximated  
- Limited dataset size  
- No real clinical validation  

---

## 🔮 Future Work
- Multi-disease classification  
- Larger dataset training  
- Integration with hospital systems  
- Real-time deployment  

---

## 👨‍💻 Authors
Deepesh Bhardwaj  
Sambhav Sharma  
Pratham Sharma  

Amity University, Noida  

---

## 📜 License
This project is for academic and research purposes only.
