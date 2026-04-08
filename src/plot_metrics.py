import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("results/training_metrics.csv")

# Loss curve
plt.figure()
plt.plot(df["epoch"], df["loss"])
plt.title("Training Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.savefig("results/loss_curve.png", dpi=300)
plt.show()


# Accuracy curve
plt.figure()
plt.plot(df["epoch"], df["accuracy"])
plt.title("Accuracy Curve")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.savefig("results/accuracy_curve.png", dpi=300)
plt.show()


# Precision curve
plt.figure()
plt.plot(df["epoch"], df["precision"])
plt.title("Precision Curve")
plt.xlabel("Epoch")
plt.ylabel("Precision")
plt.savefig("results/precision_curve.png", dpi=300)
plt.show()


# Recall curve
plt.figure()
plt.plot(df["epoch"], df["recall"])
plt.title("Recall Curve")
plt.xlabel("Epoch")
plt.ylabel("Recall")
plt.savefig("results/recall_curve.png", dpi=300)
plt.show()


# F1 curve
plt.figure()
plt.plot(df["epoch"], df["f1"])
plt.title("F1 Score Curve")
plt.xlabel("Epoch")
plt.ylabel("F1 Score")
plt.savefig("results/f1_curve.png", dpi=300)
plt.show()
