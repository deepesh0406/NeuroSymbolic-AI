"""
Neuro-Symbolic Reasoning Engine
Converts predicted medical signs into pneumonia diagnosis
"""

import numpy as np


class SymbolicReasoner:

    def __init__(self, threshold=0.65):

        self.threshold = threshold

        # Medical importance weights (based on radiology relevance)
        self.weights = {
            "opacity": 0.35,
            "consolidation": 0.30,
            "infiltration": 0.20,
            "inflammation": 0.15
        }


    def compute_score(self, sign_probs):

        opacity, consolidation, infiltration, inflammation = sign_probs

        score = (
            self.weights["opacity"] * opacity +
            self.weights["consolidation"] * consolidation +
            self.weights["infiltration"] * infiltration +
            self.weights["inflammation"] * inflammation
        )

        return score


    def predict(self, sign_probs):

        score = self.compute_score(sign_probs)

        prediction = 1 if score > self.threshold else 0

        explanation = {

            "opacity": float(sign_probs[0]),
            "consolidation": float(sign_probs[1]),
            "infiltration": float(sign_probs[2]),
            "inflammation": float(sign_probs[3]),
            "pneumonia_score": float(score),
            "threshold": float(self.threshold),
            "diagnosis": "PNEUMONIA" if prediction == 1 else "NORMAL"
        }

        return prediction, explanation


# Simple function wrapper
def symbolic_reasoning(sign_probs, threshold=0.65):

    reasoner = SymbolicReasoner(threshold)

    return reasoner.predict(sign_probs)
