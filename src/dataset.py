import os
from PIL import Image
from torch.utils.data import Dataset


class ChestXrayDataset(Dataset):

    def __init__(self, root_dir, transform=None):

        self.root_dir = root_dir
        self.transform = transform

        self.image_paths = []
        self.labels = []

        classes = ["NORMAL", "PNEUMONIA"]

        for label, class_name in enumerate(classes):

            class_dir = os.path.join(root_dir, class_name)

            if not os.path.exists(class_dir):
                raise Exception(f"Folder not found: {class_dir}")

            for img_name in os.listdir(class_dir):

                img_path = os.path.join(class_dir, img_name)

                self.image_paths.append(img_path)
                self.labels.append(label)


    def __len__(self):

        return len(self.image_paths)


    def __getitem__(self, idx):

        img_path = self.image_paths[idx]

        image = Image.open(img_path).convert("RGB")

        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label
