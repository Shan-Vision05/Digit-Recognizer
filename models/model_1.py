from torch import nn

class MNISTModel_V1(nn.Module):
  """
  Desc: Simple model with two conv layers for Feature extraction, 
  with an MLP classifier.

  Accuracy Achieved(Kaggle digit recognizer competition): 99.207%
  """
  def __init__(self, input_size, num_classes):
    super().__init__()

    self.Conv_Block_1 = nn.Sequential(
        nn.Conv2d(input_size, 32, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2)
    )
    self.Conv_Block_2 = nn.Sequential(
        nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2)
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
    x = self.Conv_Block_1(x)
    x = self.Conv_Block_2(x)
    x = self.Classifier(x)
    return x