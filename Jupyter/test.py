import torch
print(torch.hub.get_dir())
import os
print(os.getenv('TORCH_HOME'))

import torch
import torchvision
from torch import nn
from collections import OrderedDict

# 通过注释此行代码来观察方法1的缺陷
# from model_save import *

tudui_method1 = torch.load("../Models/Tudui/tudui_method1.pth")
print(tudui_method1)


