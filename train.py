import torch
from torch import optim
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader

from data.dataLoader import dataset
from models.WGAN import Discriminator, Generative

from torch.optim import RMSprop
import cv2
import os

class Config:
    """
    Some config for the model
    """
    batch_size = 128
    max_epoch = 5
    latent = 100

    extend_iter = 2
    upper_bound = 0.01

    lr = 0.00005

if __name__ == '__main__':
    config = Config()

    os.environ['CUDA_VISIBLE_DEVICES'] = '2'

    device = torch.device('cuda:0' if torch.cuda.is_available else 'cpu')
    # models
    Dis = Discriminator()
    Gen = Generative()
    # dataset
    dataloader = DataLoader(dataset, config.batch_size, shuffle=True, num_workers=2)
    # criterion
    ## Third edit: no log loss function
    # criterion = nn.BCELoss()
    # ptimizer
    ## Second edit: move the Adam momentum optimizer to the RMSprop
    optimizer_G = RMSprop(Gen.parameters(), lr=config.lr)
    optimizer_D = RMSprop(Dis.parameters(), lr=config.lr)

    for epoch in range(config.max_epoch):
        for i, data in enumerate(dataloader):
            ## Fourth Edit: clip parameter
            for param in Dis.parameters():
                param.data.clip_(-config.upper_bound, config.upper_bound)
            
            Dis.zero_grad()
            Dis.to(device)
            Gen.to(device)

            noise = torch.randn((data[0].size(0), config.latent, 1, 1), device=device)
            output = Dis(data[0].to(device)).mean()
            output.backward()
            output2 = Dis(Gen(noise).detach()).mean()
            output2.backward(torch.tensor(-1, dtype=torch.float32))

            optimizer_D.step()

            if i % 3 == 0:
                Gen.zero_grad()
                output1 = Dis(Gen(noise)).mean()
                output1.backward()
                optimizer_G.step()

            if i % 50 == 0:
                print(f"epoch:{epoch}, i:{i}. Discriminar error:{output+output2}, Generative error:{output1.cpu().item()}")
    
    torch.save(Gen, 'models.pth')
    with torch.no_grad():
        noise = torch.randn((10, 100, 1, 1), dtype=torch.float32, device=device)
        res = Gen(noise)
        for i in range(res.shape[0]):
            res[i] = res[i] * 0.5 + 0.5
            img = res[i][[2,1,0], :, :]
            img = img.permute(1,2,0)*255
            img = img.cpu().numpy().astype(np.uint8).copy()
            cv2.imwrite(f'create/{i}.jpg', img)












