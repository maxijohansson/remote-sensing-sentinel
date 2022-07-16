import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose
import matplotlib.pyplot as plt

class SentinelLandCoverSwedenDataset(Dataset):
    '''
    In the data files, the first 13 rows/arrays are the optical bands from Sentinel-L2A and the 14th is the land cover mask with 25 categories from Naturvårdsverket
    '''

    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.file_names = os.listdir(self.data_dir)

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        img = np.genfromtxt(os.path.join(self.data_dir, self.file_names[idx]), delimiter=',')
        img = img.reshape(256, 256, 14)

        sample = {'image': img[:, :, :-1], 'mask': img[:, :, -1]}

        if self.transform:
            sample = self.transform(sample)

        return sample

    
# if __name__ =='__main__':
    # sentinel_dataset = SentinelLandCoverSwedenDataset(
    #     data_dir = 'data\\SentinelLandCoverSweden\\dataset',
    #     transform = Compose([
    #         ToTensor()
    #     ])
    # )

    # for i in range(len(sentinel_dataset)):
    #     sample = sentinel_dataset[i]

    #     print(i, sample['image'].shape, sample['mask'].shape)
    
    #     if i == 3:
    #         plt.show()
    #         break