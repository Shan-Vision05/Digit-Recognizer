import torch
import pandas as pd
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from infra.DataLoader import MNISTDataLoader
from infra.MNISTDatasetModule import MNISTDataset

from models.model_1 import MNISTModel_V1
from models.model_2 import MNISTModel_V2
from models.model_3 import MNISTModel_V3

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
