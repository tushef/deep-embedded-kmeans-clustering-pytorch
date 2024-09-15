import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms


class DatasetWrapper:
    def __init__(self, dataset_name, root='data', transform=None, download=True):
        self.dataset_name = dataset_name.lower()
        self.root = root
        self.transform = transform if transform else self.default_transform()
        self.download = download
        self.train_data = None
        self.test_data = None
        self.full_data = FullDataset([],[])
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
        self.full_data.data = torch.concat([self.train_data.data, self.test_data.data], dim=0)
        self.full_data.targets = torch.concat([self.train_data.targets, self.test_data.targets], dim=0)

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
        """Return the image shape"""
        return [self.image_height, self.image_width, self.channels]

    def get_flattened_data(self, dataset):
        """Return the entire dataset with flattened images."""
        all_data = []
        all_labels = []

        # Loop through all samples in the full dataset
        for data, label in dataset:
            # Flatten the image tensor
            flattened_data = data.view(-1)  # Flatten the image tensor to 1D
            all_data.append(flattened_data)
            all_labels.append(label)

        # Stack all data and convert to a tensor
        flattened_data_tensor = torch.stack(all_data)
        labels_tensor = torch.tensor(all_labels)

        return FlattenedDataset(flattened_data_tensor, labels_tensor)

    def get_dataloader(self, dataset_type='full', batch_size=256, shuffle=True, num_workers=0):
        """
        Return a DataLoader for the specified dataset.

        Args:
            dataset_type (str): The type of dataset to load ('train', 'test', 'full').
            batch_size (int): Number of samples per batch.
            shuffle (bool): Whether to shuffle the data.
            num_workers (int): Number of subprocesses to use for data loading.

        Returns:
            DataLoader: DataLoader object for the specified dataset.
        """
        dataset = self.get_type_of_dataset(dataset_type)

        dataset = TensorDataset(dataset.data.float().unsqueeze(-1).permute(0, 3, 1, 2))

        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    def get_type_of_dataset(self, dataset_type):
        if dataset_type == 'train':
            dataset = self.get_train_data()
        elif dataset_type == 'test':
            dataset = self.get_test_data()
        elif dataset_type == 'full':
            dataset = self.get_full_data()
        else:
            raise ValueError(f"Dataset type '{dataset_type}' is not supported. Choose from 'train', 'test', or 'full'.")

        return dataset


class FullDataset:
    def __init__(self, data, target):
        self.data = data  # This will hold all the images
        self.target = target  # This will hold all the labels

    def __len__(self):
        """Return the length of the full dataset (train + test)."""
        return len(self.data)

class FlattenedDataset:
    def __init__(self, data, target):
        self.data = data  # This will hold all the images
        self.target = target  # This will hold all the labels

    def __len__(self):
        """Return the length of the dataset (train + test)."""
        return len(self.data)
