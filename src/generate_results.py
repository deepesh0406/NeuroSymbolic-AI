from src.knowledge_graph import MedicalKnowledgeGraph

def generate_report(confidence):

    kg = MedicalKnowledgeGraph()

    findings = ["PatchyOpacity", "HighActivationLowerLobe"]

    reasoning = kg.infer(findings)

    print("\nNeuro-Symbolic Diagnosis Report")
    print("------------------------------")
    print("CNN Confidence:", confidence)
    print("\nObserved Findings:", findings)
    print("\nReasoning Path:")

    for step in reasoning:
        print(f"{step[0]} --{step[1]}--> {step[2]}")


if __name__ == "__main__":
    generate_report(0.82)
