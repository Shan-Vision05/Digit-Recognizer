import torch
from torch.utils.data import Dataset
from torchvision import transforms

class MNISTDataset(Dataset):
  def __init__(self, images, labels, transform = None):
    self.images = images
    self.labels = labels
    self.transform = transform

  def __len__(self):
    return len(self.labels)

  def __getitem__(self, idx):
    image = self.images[idx]
    image = image / 255.0
    label = self.labels[idx]

    if self.transform:
      image = self.transform(image.unsqueeze(0))

    return image, label