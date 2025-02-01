import torch
from torch import nn
from tqdm.auto import tqdm
import numpy as np

class MNISTTrainer():
    def __init__(self, model, device, train_dataloader, test_dataloader):
        
        self.device = device
        self.model = model
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader

        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        # self.scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.96)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=3, gamma=0.85)
        # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    
    def accuracy_fn(self, y_true, y_pred):
        correct = torch.eq(y_true, y_pred).sum().item()
        acc = (correct / len(y_pred)) * 100
        return acc

    def train_step(self):
        self.model.train()
        train_loss = 0
        train_acc = 0
        for batch, (X, y) in enumerate(self.train_dataloader):
            X, y = X.to(self.device), y.to(self.device)

            y_pred = self.model(X)


            loss = self.loss_fn(y_pred, y)

            train_loss += loss
            # print(train_loss)
            train_acc += self.accuracy_fn(y, y_pred.argmax(dim=1))

            self.optimizer.zero_grad()

            loss.backward()

            self.optimizer.step()

        train_loss /= len(self.train_dataloader)
        train_acc /= len(self.train_dataloader)
        return train_loss, train_acc
    
    def eval_step(self):
        self.model.eval()

        with torch.inference_mode():
            test_loss = 0
            test_acc = 0
            for batch, (X, y) in enumerate(self.test_dataloader):
                X, y = X.to(self.device), y.to(self.device)

                y_pred = self.model(X)

                loss = self.loss_fn(y_pred, y)

                test_loss += loss
                # print(test_loss)
                test_acc +=  self.accuracy_fn(y, y_pred.argmax(dim=1))
            test_loss /= len(self.test_dataloader)
            test_acc /= len(self.test_dataloader)
            return test_loss, test_acc