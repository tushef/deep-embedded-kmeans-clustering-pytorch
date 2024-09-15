import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class DatasetWrapper:
    def __init__(self, dataset_name, root='data', transform=None, download=True):
        self.dataset_name = dataset_name.lower()
        self.root = root
        self.transform = transform if transform else self.default_transform()
        self.download = download
        self.train_data = None
        self.test_data = None
        self.full_data = None
        self.image_height = None
        self.image_width = None
        self.channels = None

        # Load dataset
        self.load_dataset()

    def default_transform(self):
        # Default transformation to convert images to tensor
        return transforms.Compose([transforms.ToTensor()])

    def load_dataset(self):
        """
        Load the dataset based on the dataset name and set attributes such as image size and channels.
        """
        if self.dataset_name == 'mnist':
            self.train_data = datasets.MNIST(self.root, train=True, transform=self.transform, download=self.download)
            self.test_data = datasets.MNIST(self.root, train=False, transform=self.transform, download=self.download)
        elif self.dataset_name == 'cifar10':
            self.train_data = datasets.CIFAR10(self.root, train=True, transform=self.transform, download=self.download)
            self.test_data = datasets.CIFAR10(self.root, train=False, transform=self.transform, download=self.download)
        elif self.dataset_name == 'fashionmnist':
            self.train_data = datasets.FashionMNIST(self.root, train=True, transform=self.transform,
                                                    download=self.download)
            self.test_data = datasets.FashionMNIST(self.root, train=False, transform=self.transform,
                                                   download=self.download)
        elif self.dataset_name == 'kmnist':
            self.train_data = datasets.KMNIST(self.root, train=True, transform=self.transform,
                                              download=self.download)
            self.test_data = datasets.KMNIST(self.root, train=False, transform=self.transform,
                                             download=self.download)
        elif self.dataset_name == 'stl10':
            self.train_data = datasets.STL10(self.root, train=True, transform=self.transform,
                                             download=self.download)
            self.test_data = datasets.STL10(self.root, train=False, transform=self.transform,
                                            download=self.download)
        elif self.dataset_name == 'usps':
            self.train_data = datasets.USPS(self.root, train=True, transform=self.transform,
                                            download=self.download)
            self.test_data = datasets.USPS(self.root, train=False, transform=self.transform,
                                           download=self.download)
        # Add more datasets here as needed

        else:
            raise ValueError(f"Dataset '{self.dataset_name}' is not supported.")

        # Get full dataset (train + test)
        self.create_full_dataset()

        # Set image size and channels from the first item in the train dataset
        self.image_height, self.image_width = self.train_data[0][0].shape[1:]
        self.channels = self.train_data[0][0].shape[0]

    def create_full_dataset(self):
        self.full_data = torch.utils.data.ConcatDataset([self.train_data, self.test_data])
        # Lists to collect all data and labels
        all_data = []
        all_labels = []

        for i in range(len(self.full_data)):
            data, label = self.full_data[i]
            all_data.append(data)
            all_labels.append(label)

        # Convert lists to tensors
        full_data_tensor = torch.stack(all_data)  # Stack the data (images)
        full_labels_tensor = torch.tensor(all_labels)  # Convert labels to tensor
        self.full_data = FullDataset(full_data_tensor, full_labels_tensor)

    def get_train_data(self):
        """Return the training dataset."""
        return self.train_data

    def get_test_data(self):
        """Return the testing dataset."""
        return self.test_data

    def get_full_data(self):
        """Return the full dataset (train + test)."""
        return self.full_data

    def get_image_shape(self):
        """Return the image shape as (channels, height, width)."""
        return (self.channels, self.image_height, self.image_width)


class FullDataset:
    def __init__(self, data, target):
        self.data = data  # This will hold all the images
        self.target = target  # This will hold all the labels
