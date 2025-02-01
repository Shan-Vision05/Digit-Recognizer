from torch import nn

class ResidualBlock(nn.Module):
  def __init__(self, size):
    super().__init__()

    self.Conv_Block_1 = nn.Sequential(
        nn.Conv2d(size, size, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(size),
        nn.ReLU()
    )

    self.Conv_Block_2 = nn.Sequential(
        nn.Conv2d(size, size, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(size),
        nn.ReLU()
    )

  def forward(self, x):
    cb1 = self.Conv_Block_1(x)
    cb2 = self.Conv_Block_2(cb1)
    return x + cb2