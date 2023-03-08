from torch import nn

class CIFAR10Classifier(nn.Module):
  def __init__(self, input_shape: int,
               output_shape: int,
               hidden_units: int):
    super().__init__()
    self.cnn_block_1 = nn.Sequential(
        nn.Conv2d(in_channels=input_shape,
                  out_channels=hidden_units,
                  kernel_size=3,
                  stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2)
    )
    self.cnn_block_2 = nn.Sequential(
        nn.Conv2d(in_channels=hidden_units,
                  out_channels=hidden_units*2,
                  kernel_size=3,
                  stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2)
    )
    self.classifier = nn.Sequential(
        nn.Flatten(),
        nn.Linear(in_features= hidden_units*8*8*2, out_features=hidden_units*8*2),
        nn.ReLU(),
        nn.Linear(in_features= hidden_units*8*2, out_features=output_shape))
    
  def forward(self, x):
    x = self.cnn_block_1(x)
    x = self.cnn_block_2(x)
    x = self.classifier(x)
    return x