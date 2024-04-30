import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
from PIL import Image

__all__= ['CIFAR_New']

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

cifar_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

class CIFAR_New(data.Dataset):
	def __init__(self, root = '/mnt/cephfs/dataset/TTA/CIFAR-10.1/datasets', transform=cifar_transform, target_transform=None, version='v6'):
		self.data = np.load('%s/cifar10.1_%s_data.npy' %(root, version))
		self.targets = np.load('%s/cifar10.1_%s_labels.npy' %(root, version)).astype('long')
		self.transform = transform
		self.target_transform = target_transform

	def __getitem__(self, index):
		img, target = self.data[index], self.targets[index]
		img = Image.fromarray(img)
		if self.transform is not None:
			img = self.transform(img)
		if self.target_transform is not None:
			target = self.target_transform(target)
		return img, target

	def __len__(self):
		return len(self.targets)