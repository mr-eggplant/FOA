import numpy as np
import torchvision
import torch
import torch.utils.data

def prepare_dataset(dataroot,
                    corruption,
                    level,
                    batch_size,
                    num_workers,
                    te_transforms):
    tesize = 10000
    print(f'Test on {corruption} level {level}')
    teset_raw = np.load(dataroot + '/%s.npy' %(corruption))
    teset_raw = teset_raw[(level-1)*tesize: level*tesize]
    teset = torchvision.datasets.CIFAR10(root=dataroot,
                                        train=False, download=True, transform=te_transforms)
    teset.data = teset_raw

    teloader = torch.utils.data.DataLoader(teset, 
                                           batch_size=batch_size, 
                                           shuffle=False,
                                           num_workers=num_workers,
                                           pin_memory=True)
    return teloader
    
