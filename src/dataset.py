import os
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class ChestXrayDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.image_paths = []
        self.labels = []
        self.transform = transform

        for label_name in ["NORMAL", "PNEUMONIA"]:
            label_dir = os.path.join(root_dir, label_name)
            label = 0 if label_name == "NORMAL" else 1

            for img in os.listdir(label_dir):
                if img.lower().endswith((".png", ".jpg", ".jpeg")):
                    self.image_paths.append(os.path.join(label_dir, img))
                    self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label
