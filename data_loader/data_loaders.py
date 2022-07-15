from torchvision import datasets, transforms
from base import BaseDataLoader


class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.data_dir = data_dir
        self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

class SentinelDataLoader(BaseDataLoader):
    '''
    Loads the images from ../data/. each image is an ndarray with shape 14x512x512. 
    The first 13 arrays are the optical bands from Sentinel-L2A and the 14th is the land cover mask with 25 categories from Naturvårdsverket
    '''