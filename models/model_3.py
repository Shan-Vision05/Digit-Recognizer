import torch
from torch import nn
from .ResidualBlockModule import ResidualBlock


class MNISTModel_V3(nn.Module):
  """
  Desc: Improved on Model_2 by adding ResNets to multiple different feature outputs,
    from different Convolution strides to capture different info and similar MLP Classifier Head

  Accuracy Achieved(Test Accuracy): 99.550%
  """

  def __init__(self, input_size, num_classes):
    super().__init__()
    self.channels = 128

    # (64, 128, 28, 28)
    self.Conv_Block = nn.Sequential(
        nn.Conv2d(input_size, self.channels, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(self.channels),
        nn.ReLU()
    )
    # (64, 128, 28, 28)



    # (64, 128, 28, 28)
    self.Conv_1 = nn.Sequential(
        nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1),
        nn.BatchNorm2d(self.channels),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2)
    )
    # (64, 128, 13, 13)

    # (64, 128, 28, 28)
    self.Conv_2 = nn.Sequential(
        nn.Conv2d(self.channels, self.channels, kernel_size=5, stride=1),
        nn.BatchNorm2d(self.channels),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2)
    )
    # (64, 128, 12, 12)

    # (64, 128, 28, 28)
    self.Conv_3 = nn.Sequential(
        nn.Conv2d(self.channels, self.channels, kernel_size=7, stride=1),
        nn.BatchNorm2d(self.channels),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2)
    )
    # (64, 128, 11, 11)

    self.Conv_4 = nn.Sequential(
        nn.Conv2d(self.channels, self.channels, kernel_size=2, stride=1),
        nn.BatchNorm2d(self.channels),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2)
    )


    self.ResNet_1 = nn.Sequential(
        *[ResidualBlock(self.channels) for _ in range(3)]
    )
    # (64, 128, 13, 13)

    self.ResNet_2 = nn.Sequential(
        *[ResidualBlock(self.channels) for _ in range(3)]
    )
    # (64, 128, 12, 12)

    self.ResNet_3 = nn.Sequential(
        *[ResidualBlock(self.channels) for _ in range(3)]
    )
    # (64, 128, 11, 11)
    self.ResNet_4 = nn.Sequential(
        *[ResidualBlock(self.channels) for _ in range(3)]
    )
    # (64, 128, 11, 11)


    self.Conv_End_1 = nn.Sequential(
        nn.Conv2d(self.channels, 64 , kernel_size=3, stride=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        nn.Flatten(),
    )
    # (64, 128, 7, 7)

    self.Conv_End_2 = nn.Sequential(
        nn.Conv2d(self.channels, 64 , kernel_size=3, stride=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2),
        nn.Flatten(),
    )
    # (64, 128, 6, 6)

    self.Conv_End_3 = nn.Sequential(
        nn.Conv2d(self.channels, 64 , kernel_size=3, stride=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        nn.Flatten()
    )
    # (64, 128, 6, 6)

    self.Conv_End_4 = nn.Sequential(
        nn.Conv2d(self.channels, 64 , kernel_size=3, stride=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        nn.Flatten()
    )
    # (64, 128, 6, 6)

    self.Classifier = nn.Sequential(
        nn.Linear(5952, 1568),
        nn.Dropout(0.5),
        nn.ReLU(),
        nn.Linear(1568, 256),
        nn.Dropout(0.5),
        nn.ReLU(),
        nn.Linear(256, num_classes)
    )

  def forward(self, x):
    cb1 = self.Conv_Block(x)

    cv1 = self.Conv_1(cb1)
    cv2 = self.Conv_2(cb1)
    cv3 = self.Conv_3(cb1)
    cv4 = self.Conv_4(cb1)

    rn1 = self.ResNet_1(cv1)
    rn2 = self.ResNet_2(cv2)
    rn3 = self.ResNet_3(cv3)
    rn4 = self.ResNet_4(cv4)


    cve1 = self.Conv_End_1(rn1)
    cve2 = self.Conv_End_2(rn2)
    cve3 = self.Conv_End_3(rn3)
    cve4 = self.Conv_End_4(rn4)

    concatenated = torch.cat((cve1, cve2, cve3, cve4), dim=1)

    end = self.Classifier(concatenated)
    return end