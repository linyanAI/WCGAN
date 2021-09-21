import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

class Config:
    latent  = 100
    base_channel = 128

    mean = 0
    std = 0.02

config = Config()

class Generative(nn.Module):
    # 1. ConvTranspose2d的原理？
    def __init__(self):
        super(Generative, self).__init__()
        self.args = OrderedDict([
            ('deconv1', nn.ConvTranspose2d(config.latent,8*config.base_channel,4,1, 0, bias=False)),
            ('batchnorm1', nn.BatchNorm2d(8*config.base_channel)),
            ('activate1', nn.ReLU(True)),
            ('deconv2', nn.ConvTranspose2d(8*config.base_channel,4*config.base_channel,4,2,1, bias=False)),
            ('batchnorm2', nn.BatchNorm2d(4*config.base_channel)),
            ('activate2', nn.ReLU(True)),
            ('deconv3', nn.ConvTranspose2d(4*config.base_channel,2*config.base_channel,4,2,1, bias=False)),
            ('batchnorm3', nn.BatchNorm2d(2*config.base_channel)),
            ('activate3', nn.ReLU(True)),
            ('deconv4', nn.ConvTranspose2d(2*config.base_channel, config.base_channel,4,2,1, bias=False)),
            ('batchnorm4', nn.BatchNorm2d(config.base_channel)),
            ('activate4', nn.ReLU(True)),
            ('deconv5', nn.ConvTranspose2d(config.base_channel,3,4,2,1, bias=False)),
            ('activate5', nn.Tanh()),
        ])
        self.network = nn.Sequential(self.args)
        self.network.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.ConvTranspose2d):
            nn.init.normal_(m.weight.data, mean=config.mean, std=config.std)
        if isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight.data, mean=config.mean, std=config.std)

    def forward(self,  x):
        x = self.network(x)

        return x


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.args = OrderedDict([
            ('Conv1', nn.Conv2d(3, config.base_channel, 4, 2, 1, bias=False)),
            ('Activate1', nn.LeakyReLU(negative_slope=0.2,inplace=True)),
            ('Conv2', nn.Conv2d(config.base_channel, 2*config.base_channel, 4, 2, 1, bias=False)),
            ('BatchNorm2', nn.BatchNorm2d(2*config.base_channel)),
            ('Activate2', nn.LeakyReLU(negative_slope=0.2,inplace=True)),
            ('Conv3', nn.Conv2d(2*config.base_channel, 4*config.base_channel, 4, 2, 1, bias=False)),
            ('BatchNorm3', nn.BatchNorm2d(4*config.base_channel)),
            ('Activate3', nn.LeakyReLU(negative_slope=0.2,inplace=True)),
            ('Conv4', nn.Conv2d(4*config.base_channel, 8*config.base_channel, 4, 2, 1, bias=False)),
            ('BatchNorm4', nn.BatchNorm2d(8*config.base_channel)),
            ('Activate4', nn.LeakyReLU(negative_slope=0.2,inplace=True)),
            ('Conv5', nn.Conv2d(8*config.base_channel, 1, 4, 1, 0)),
            ## first edit: reomove the activate functions sigmoid
            # ('Activate5', nn.Sigmoid()),
        ])
        self.network = nn.Sequential(self.args)

        self.network.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.ConvTranspose2d):
            nn.init.normal_(m.weight.data, mean=config.mean, std=config.std)
        if isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight.data, mean=config.mean, std=config.std)
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight.data, mean=config.mean, std=config.std)

    def forward(self, x):
        x = self.network(x)

        return x



if __name__ == "__main__":
    G = Generative()
    D = Discriminator()
    x = torch.randn((1, config.latent, 1, 1))
    print(x.shape)
    output = G(x)
    print(output.shape)
    result = D(output)
    print(result.shape)