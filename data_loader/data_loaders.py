import sys
import os
import torch
from torchvision import datasets, transforms
from torchvision.transforms import Compose
import matplotlib.pyplot as plt

sys.path.insert(1, os.path.join(sys.path[0], '..\\..'))
from data_acquisition.data_utils import plot_image_mask_batch
from data_loader.datasets import SentinelLandCoverSwedenDataset

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'mask': torch.from_numpy(mask)}


if __name__ == '__main__':
    transformed_dataset = SentinelLandCoverSwedenDataset(
        data_dir = 'data\\SentinelLandCoverSweden\\dataset',
        transform = Compose([
            ToTensor()
        ])
    )

    data_loader = torch.utils.data.DataLoader(dataset=transformed_dataset, batch_size=4, shuffle=True, num_workers=0)

    for i_batch, batch in enumerate(data_loader):
        print(i_batch, batch['image'].size(),
            batch['mask'].size())

        # observe 4th batch and stop.
        if i_batch == 0:
            plot_image_mask_batch(batch)
            # plt.axis('off')
            # plt.ioff()
            plt.show()
            break