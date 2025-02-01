import torch
import pandas as pd
import matplotlib.pyplot as plt

from tqdm.auto import tqdm
import numpy as np

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from infra.DataLoader import MNISTDataLoader
from infra.MNISTDatasetModule import MNISTDataset

from models.model_1 import MNISTModel_V1
from models.model_2 import MNISTModel_V2
from models.model_3 import MNISTModel_V3

from Trainer import MNISTTrainer

df = pd.read_csv('data/train.csv')

transform = transforms.Compose([
    transforms.Normalize((0.1310,), (0.3085,))
])
mnistDataLoader = MNISTDataLoader()
train_loader, test_loader = mnistDataLoader.DataLoader(df, transform)

# Display image and label.
train_features, train_labels = next(iter(train_loader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")
img = train_features[5].squeeze()
label = train_labels[5]
plt.imshow(img, cmap="gray")
plt.show()
print(f"Label: {label}")

device = "cuda" if torch.cuda.is_available() else "cpu"


model1 = MNISTModel_V1(1, 10)
model1 = model1.to(device)

model2 = MNISTModel_V2(1, 10)
model2 = model2.to(device)

model3 = MNISTModel_V3(1, 10)
model3 = model3.to(device)

import torchinfo
torchinfo.summary(model=model3,
        input_size=(64, 1, 28, 28), # make sure this is "input_size", not "input_shape"
        # col_names=["input_size"], # uncomment for smaller output
        col_names=["input_size", "output_size", "num_params", "trainable"],
        col_width=20,
        row_settings=["var_names"]
)

## Training and Eval Part

epochs = 100
best_val_loss = np.inf
best_model = None

model = model1
trainer = MNISTTrainer(model, device, train_loader, test_loader)

num_epochs_without_improvement = 0
patience = 5
for epoch in tqdm(range(epochs)):

  train_loss, train_acc = trainer.train_step()
  trainer.scheduler.step()

  test_loss, test_acc = trainer.eval_step()

  if test_loss < best_val_loss:
    best_val_loss = test_loss
    best_model = model.state_dict()
    num_epochs_without_improvement = 0
  else:
    num_epochs_without_improvement += 1

  print(f'Epoch {epoch}, Train Loss: {train_loss:.3f}, Test Loss: {test_loss:.3f} Train Acc: {train_acc:.3f}, Test Acc: {test_acc:.3f}')
  if num_epochs_without_improvement >= patience:
    print("Early Stopping Triggered!!!")
    break