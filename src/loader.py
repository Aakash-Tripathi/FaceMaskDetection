"""
Loader Package
-------------------------
This package loads all the required data
"""

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import matplotlib.image as img
import os
from sklearn.model_selection import train_test_split
import pandas as pd
from torchvision import transforms


class FMDDataset(Dataset):
    """[summary]

    Args:
        Dataset ([type]): [description]
    """

    def __init__(self, data, path, transform=None):
        """[summary]

        Args:
            data ([type]): [description]
            path ([type]): [description]
            transform ([type], optional): [description]. Defaults to None.
        """
        super().__init__()
        self.data = data.values
        self.path = path
        self.transform = transform

    def __len__(self):
        """[summary]

        Returns:
            [type]: [description]
        """
        return len(self.data)

    def __getitem__(self, index):
        """[summary]

        Args:
            index ([type]): [description]

        Returns:
            [type]: [description]
        """
        img_name, label = self.data[index]
        img_path = os.path.join(self.path, img_name)
        image = img.imread(img_path)
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def load_data(batch_size, test_size):
    """[summary]

    Args:
        batch_size ([type]): [description]
        test_size ([type]): [description]

    Returns:
        [type]: [description]
    """

    CWDpath = os.getcwd()
    path = (CWDpath + r'/data/')
    labels = pd.read_csv(path+r'train.csv')
    train_path = path+r'FMD/'

    transformer = transforms.Compose([transforms.ToPILImage(),
                                      transforms.ToTensor(),
                                      transforms.Resize((256, 256)),
                                      transforms.RandomHorizontalFlip(p=0.5),
                                      transforms.GaussianBlur(
                                          1, sigma=(0.1, 2.0)),
                                      transforms.Normalize(0, 1)])

    train, valid = train_test_split(labels, test_size=test_size)

    train_dataset = FMDDataset(train, train_path, transformer)
    test_dataset = FMDDataset(valid, train_path, transformer)

    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True)
    test_dataloader = DataLoader(dataset=test_dataset,
                                 batch_size=batch_size,
                                 shuffle=True)

    return train_dataloader, test_dataloader
