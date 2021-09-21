import torch
from torch._C import dtype
import torch.nn as nn
import numpy as np
import cv2


if __name__ == '__main__':
    device = 'cuda:0' if torch.cuda.is_available else 'cpu'
    model = torch.load('models.pth')
    noise = torch.randn((10, 100, 1, 1), dtype=torch.float32, device=device)

    model.eval()
    with torch.no_grad():
        res = model(noise)
        for i in range(res.shape[0]):
            res[i] = res[i] * 0.5 + 0.5
            img = res[i][[2,1,0], :, :]
            img = img.permute(1,2,0)*255
            img = img.cpu().numpy().astype(np.uint8).copy()
            cv2.imwrite(f'{i}.jpg', img)


