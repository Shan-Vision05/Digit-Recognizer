from torch import nn
from ResidualBlock import ResidualBlock

class MNISTModel_V2(nn.Module):
  """
  Desc: Improved on Model_1 by adding ResNets, and similar MLP Classifier Head

  Accuracy Achieved(Test Accuracy): 99.207%
  """
  def __init__(self, input_size, num_classes):
    super().__init__()
    self.channels = 128

    self.Conv_Block = nn.Sequential(
        nn.Conv2d(input_size, self.channels, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(self.channels),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2)
    )

    self.ResNet = nn.Sequential(
        *[ResidualBlock(self.channels) for _ in range(3)]
    )

    self.Conv_Block_End = nn.Sequential(
        nn.Conv2d(self.channels, 64 , kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )

    self.Classifier = nn.Sequential(
        nn.Flatten(),
        nn.Linear(64 * 7 * 7, 1568),
        nn.Dropout(0.5),
        nn.ReLU(),
        nn.Linear(1568, 256),
        nn.Dropout(0.5),
        nn.ReLU(),
        nn.Linear(256, num_classes)
    )

  def forward(self, x):
    x = self.Conv_Block(x)
    x = self.ResNet(x)
    x = self.Conv_Block_End(x)
    x = self.Classifier(x)
    return x