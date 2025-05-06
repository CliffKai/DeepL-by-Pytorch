from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

train_dataset = datasets.CIFAR10(root='../../datasets/CIFAR10', train=True, download=True, transform=transforms.ToTensor())
test_dataset  = datasets.CIFAR10(root='../../datasets/CIFAR10', train=False, download=True, transform=transforms.ToTensor())

train_data = DataLoader(dataset = train_dataset, batch_size = 1, shuffle = True, num_workers = 0, drop_last = False)

for datas in train_data:
    img, label = datas
    print(img.shape)
