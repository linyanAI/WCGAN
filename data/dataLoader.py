import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms

config={
    'root': '/home/linyan/project/dataset',
    'size': 64,
}


dataset = datasets.ImageFolder(root=config['root'], 
                                transform=transforms.Compose([
                                    transforms.Resize(config['size']),
                                    transforms.CenterCrop(config['size']),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                ]))

if __name__ == '__main__':
    data = dataset()
    print(data[0].shape)
