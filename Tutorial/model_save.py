from collections import OrderedDict
from torch import nn

class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.model1 = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, padding=2)),
            ('pool1', nn.MaxPool2d(kernel_size=2)),
            ('conv2', nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, padding=2)),
            ('pool2', nn.MaxPool2d(kernel_size=2)),
            ('conv3', nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2)),
            ('pool3', nn.MaxPool2d(kernel_size=2)),
            ('flatten', nn.Flatten()),
            ('fc1', nn.Linear(1024, 64)),  # 注意：1024 = 64通道 × 4 × 4（针对输入32x32）
            ('fc2', nn.Linear(64, 10))
        ]))

    def forward(self, x):
        return self.model1(x)