import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import pytorch_lightning as pl

from modules.dataset import MiniFASNetLightning, SpoofDataset

transform = transforms.Compose([transforms.Resize((80, 80)), transforms.ToTensor()])

dataset = SpoofDataset("dataset", transform=transform)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=32)

model = MiniFASNetLightning()

trainer = pl.Trainer(max_epochs=10)
trainer.fit(model, train_loader, val_loader)

# Save only the model weights
torch.save(model.model.state_dict(), "minifasnet_finetuned.pth")
