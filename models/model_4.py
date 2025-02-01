import torch
from torch import nn
from .ResidualBlockModule import ResidualBlock


class MNISTModel_V4(nn.Module):
  """
  Desc:  using a combination of ResNets and HRNet to capture different features at different strides 
  and similar MLP Classifier Head

  Accuracy Achieved(Test Accuracy): 99.467%
  """
  def __init__(self, input_size, num_classes):
    super().__init__()
    self.channels = 128

    # (64, 1, 28, 28)
    self.Conv_00 = nn.Sequential(
        nn.Conv2d(input_size, self.channels, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(self.channels),
        nn.ReLU(),
    )
    # (64, 128, 28, 28)

    self.ResNet_00 = ResidualBlock(self.channels)

    # (64, 128, 28, 28)
    self.Conv_11 = nn.Sequential(
        nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1),
        nn.BatchNorm2d(self.channels),
        nn.ReLU(),
    )
    # (64, 128, 26, 26)

    self.ResNet_01 = ResidualBlock(self.channels)

    self.Conv_21 = nn.Sequential(
        nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1),
        nn.BatchNorm2d(self.channels),
        nn.ReLU(),
    )

    self.Conv_22 = nn.Sequential(
        nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1),
        nn.BatchNorm2d(self.channels),
        nn.ReLU(),
    )

    self.TConv_21 = nn.Sequential(
        nn.ConvTranspose2d(self.channels, self.channels, kernel_size=3, stride=1),
        nn.BatchNorm2d(self.channels),
        nn.ReLU(),
    )

    self.Conv_32 = nn.Sequential(
        nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1),
        nn.BatchNorm2d(self.channels),
        nn.ReLU(),
    )

    self.ResNet_10 = ResidualBlock(self.channels)

    self.ResNet_11 = ResidualBlock(self.channels)

    self.ResNet_12 = ResidualBlock(self.channels)

    self.Conv_41 = nn.Sequential(
        nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1),
        nn.BatchNorm2d(self.channels),
        nn.ReLU(),
    )

    self.Conv_42 = nn.Sequential(
        nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1),
        nn.BatchNorm2d(self.channels),
        nn.ReLU(),
    )

    self.Conv_43 = nn.Sequential(
        nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1),
        nn.BatchNorm2d(self.channels),
        nn.ReLU(),
    )

    self.TConv_41 = nn.Sequential(
        nn.ConvTranspose2d(self.channels, self.channels, kernel_size=3, stride=1),
        nn.BatchNorm2d(self.channels),
        nn.ReLU(),
    )

    self.TConv_42 = nn.Sequential(
        nn.ConvTranspose2d(self.channels, self.channels, kernel_size=3, stride=1),
        nn.BatchNorm2d(self.channels),
        nn.ReLU(),
    )

    self.TConv_43 = nn.Sequential(
        nn.ConvTranspose2d(self.channels, self.channels, kernel_size=3, stride=1),
        nn.BatchNorm2d(self.channels),
        nn.ReLU(),
    )

    self.MaxPool_Flatten_00 = nn.Sequential(
        nn.MaxPool2d(kernel_size=3, stride=2),
        nn.Flatten()
    )

    self.MaxPool_Flatten_01 = nn.Sequential(
        nn.MaxPool2d(kernel_size=3, stride=2),
        nn.Flatten()
    )

    self.MaxPool_Flatten_02 = nn.Sequential(
        nn.MaxPool2d(kernel_size=3, stride=2),
        nn.Flatten()
    )

    self.Classifier = nn.Sequential(
        # nn.Flatten(),
        nn.Linear(128*13*13 + 128*12*12 + 128*11*11, 4000),
        nn.Dropout(0.5),
        nn.ReLU(),
        nn.Linear(4000, 2000),
        nn.Dropout(0.5),
        nn.ReLU(),
        nn.Linear(2000, 1000),
        nn.Dropout(0.5),
        nn.ReLU(),
        nn.Linear(1000, 256),
        nn.Dropout(0.5),
        nn.ReLU(),
        nn.Linear(256, num_classes)
    )

  def forward(self, x):

    out_Conv_00 = self.Conv_00(x)

    out_ResNet_00 = self.ResNet_00(out_Conv_00)
    out_Conv_11 = self.Conv_11(out_Conv_00)
    out_ResNet_01 = self.ResNet_01(out_Conv_11)

    out_Conv_21 = self.Conv_21(out_ResNet_00)
    out_Conv_32 = self.Conv_32(out_Conv_21)

    out_Conv_22 = self.Conv_22(out_ResNet_01)
    out_TConv_21 = self.TConv_21(out_ResNet_01)

    out_ResNet_10 = self.ResNet_10(out_ResNet_00 + out_TConv_21)
    out_ResNet_11 = self.ResNet_11(out_Conv_21 + out_ResNet_01)
    out_ResNet_12 = self.ResNet_12(out_Conv_32 + out_Conv_22)

    out_Conv_41 = self.Conv_41(out_ResNet_10)
    out_Conv_42 = self.Conv_42(out_ResNet_11)
    out_Conv_43 = self.Conv_43(out_Conv_41)

    out_TConv_41 = self.TConv_41(out_ResNet_11)
    out_TConv_42 = self.TConv_42(out_ResNet_12)
    out_TConv_43 = self.TConv_43(out_TConv_42)

    out_layer_1 = out_ResNet_10 + out_TConv_41 + out_TConv_43
    out_layer_2 = out_ResNet_11 + out_Conv_41 + out_TConv_42
    out_layer_3 = out_ResNet_12 + out_Conv_42 + out_Conv_43

    out_MaxPool_00 = self.MaxPool_Flatten_00(out_layer_1)
    out_MaxPool_01 = self.MaxPool_Flatten_01(out_layer_2)
    out_MaxPool_02 = self.MaxPool_Flatten_02(out_layer_3)

    concatenated = torch.cat((out_MaxPool_00, out_MaxPool_01, out_MaxPool_02), dim=1)


    out = self.Classifier(concatenated)
    return out