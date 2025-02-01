## Importing Libraries

import pandas as pd

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from infra.MNISTDatasetModule import MNISTDataset

class MNISTDataLoader():

    def DataLoader(self, dataFrame, transform, batch_size = 64, train_test_ratio = 0.8):
        labels = []
        images = []
        dataSize = len(dataFrame)
        for rowId in range(dataSize):
            row = list(dataFrame.iloc[rowId])
            label = row[0]
            data = row[1:]
            img = torch.tensor(data, dtype=torch.float32).reshape(28,28)
            labels.append(torch.tensor(label, dtype = torch.int64))
            images.append(img)

        train_dataset = MNISTDataset(images[:int(dataSize*train_test_ratio)], labels[:int(dataSize*train_test_ratio)], transform)
        test_dataset = MNISTDataset(images[int(dataSize*train_test_ratio):], labels[int(dataSize*train_test_ratio):], transform)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, test_loader