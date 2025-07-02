import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import pytorch_lightning as pl

from modules.dataset import (
    MiniFASNetFineTuningLightning,
    MiniFASNetLightning,
    SpoofDataset,
)


def train_anti_spoof_model(
    dataset_path="dataset",
    output_path="models/anti_spoofing/minifasnet_custom_data.pth",
    fine_tune=True,
    max_epochs=10,
    batch_size=32,
):
    print("🔧 Starting training on spoof dataset...")

    transform = transforms.Compose(
        [
            transforms.Resize((80, 80)),
            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )

    dataset = SpoofDataset(dataset_path, transform=transform)
    if len(dataset) < 2:
        print("❌ Not enough data to train. Please capture more real/fake samples.")
        return

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    model = MiniFASNetFineTuningLightning() if fine_tune else MiniFASNetLightning()

    trainer = pl.Trainer(max_epochs=max_epochs)
    trainer.fit(model, train_loader, val_loader)

    torch.save(model.model.state_dict(), output_path)
    print(f"✅ Training complete. Weights saved to: {output_path}")
