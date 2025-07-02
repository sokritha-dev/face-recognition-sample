from torch.utils.data import Dataset
from PIL import Image
import os
import pytorch_lightning as pl
import torch
from models.mini_fasnet import MiniFASNetV1SE  # Use your model


class SpoofDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.transform = transform
        self.samples = []
        self.label_map = {"fake": 0, "real": 1}
        for label in ["fake", "real"]:
            class_dir = os.path.join(root_dir, label)
            for fname in os.listdir(class_dir):
                if fname.endswith(".jpg"):
                    self.samples.append(
                        (os.path.join(class_dir, fname), self.label_map[label])
                    )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


class MiniFASNetLightning(pl.LightningModule):
    def __init__(self, learning_rate=1e-3):
        super().__init__()
        self.model = MiniFASNetV1SE(conv6_kernel=(5, 5), num_classes=2)
        self.lr = learning_rate
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        acc = (logits.argmax(1) == y).float().mean()
        self.log("train_loss", loss)
        self.log("train_acc", acc)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        acc = (logits.argmax(1) == y).float().mean()
        self.log("val_loss", loss)
        self.log("val_acc", acc)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
