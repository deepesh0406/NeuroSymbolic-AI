import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from src.dataset import ChestXrayDataset

# Load dataset
dataset = ChestXrayDataset("data/chest_xray/train")

# Pick 8 samples
samples = [dataset[i] for i in range(8)]
images = [x[0] for x in samples]
labels = [x[1] for x in samples]

# Convert numeric labels to text
label_map = {0: "NORMAL", 1: "PNEUMONIA"}
label_names = [label_map[l] for l in labels]

# Create image grid
grid = make_grid(images, nrow=4, normalize=True)

# Plot
plt.figure(figsize=(10, 5))
plt.imshow(grid.permute(1, 2, 0))
plt.axis("off")
plt.title("Sample Chest X-Ray Images (After Preprocessing)")

# Add label text
for i, label in enumerate(label_names):
    row = i // 4
    col = i % 4
    plt.text(
        col * 128 + 10,
        row * 128 + 15,
        label,
        color="yellow",
        fontsize=10,
        weight="bold"
    )

plt.tight_layout()
plt.show()
