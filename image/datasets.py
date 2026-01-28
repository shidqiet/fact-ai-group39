import torch
import torchvision
from torch.utils.data import Dataset
from torchvision import datasets, transforms


class MNIST(Dataset):
    """Wrapper class that loads MNIST onto the GPU for speed reasons."""

    def __init__(self, train=True, download=True, device="cuda"):
        dataset = datasets.MNIST(root="./data", train=train, download=download)
        self.x = dataset.data.float().to(device).unsqueeze(1) / 255.0
        self.y = dataset.targets.to(device)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.x.size(0)


class FMNIST(Dataset):
    """Wrapper class that loads F-MNIST onto the GPU for speed reasons."""

    def __init__(self, train=True, download=True, device="cuda"):
        dataset = datasets.FashionMNIST(root="./data", train=train, download=download)
        self.x = dataset.data.float().to(device).unsqueeze(1) / 255.0
        self.y = dataset.targets.to(device)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.x.size(0)


class SVHN(Dataset):
    def __init__(
        self,
        split: str = "train",
        download: bool = True,
        device: str = "cuda",
        grayscale: bool = False,
    ):
        transform_list = []
        if grayscale:
            transform_list.append(transforms.Grayscale(num_output_channels=1))
        transform_list.append(transforms.ToTensor())
        transform = transforms.Compose(transform_list)

        dataset = datasets.SVHN(
            root="./data",
            split=split,
            download=download,
            transform=transform,
        )

        channels = 1 if grayscale else 3
        self.x = torch.empty(len(dataset), channels, 32, 32, device=device)
        self.y = torch.empty(len(dataset), dtype=torch.long, device=device)

        for i, (img, label) in enumerate(dataset):
            self.x[i] = img
            self.y[i] = label

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.x.size(0)


class CIFAR10(Dataset):
    def __init__(
        self,
        train: bool = True,
        download: bool = True,
        device: str = "cuda",
        grayscale: bool = False,
        labels: list[int] | None = None,
    ):
        transform_list = []
        if grayscale:
            transform_list.append(transforms.Grayscale(num_output_channels=1))
        transform_list.append(transforms.ToTensor())
        transform = transforms.Compose(transform_list)

        dataset = datasets.CIFAR10(
            root="./data",
            train=train,
            download=download,
            transform=transform,
        )

        channels = 1 if grayscale else 3
        if labels is None:
            indices = range(len(dataset))
        else:
            labels_set = set(labels)
            indices = [i for i, (_, y) in enumerate(dataset) if y in labels_set]

        self.x = torch.empty(len(indices), channels, 32, 32, device=device)
        self.y = torch.empty(len(indices), dtype=torch.long, device=device)
        for j, i in enumerate(indices):
            img, label = dataset[i]
            self.x[j] = img
            self.y[j] = label

        if labels is not None:
            label_map = {old: new for new, old in enumerate(labels)}
            self.y = torch.tensor(
                [label_map[int(label)] for label in self.y], device=device
            )

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.x.size(0)


class EMNIST(Dataset):
    def __init__(self, train=True, download=True, split="digits", device="cuda"):
        transform = torchvision.transforms.Compose(
            [
                lambda img: torchvision.transforms.functional.rotate(img, -90),
                lambda img: torchvision.transforms.functional.hflip(img),
                torchvision.transforms.ToTensor(),
            ]
        )
        dataset = datasets.EMNIST(
            root="./data",
            train=train,
            split=split,
            download=download,
            transform=transform,
        )
        self.x = torch.empty(len(dataset), 1, 28, 28, device=device)
        self.y = torch.empty(len(dataset), dtype=torch.long, device=device)
        for i, (img, label) in enumerate(dataset):
            self.x[i] = img
            self.y[i] = label
        if split == "letters":
            self.y -= 1

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.x.size(0)
