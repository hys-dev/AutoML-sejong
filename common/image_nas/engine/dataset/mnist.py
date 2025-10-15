import nni
from torchvision import transforms
from torchvision.datasets import MNIST, FashionMNIST
from engine.dataset.transformers import cutout_transform
from common.config import DATA_STORAGE_PATH


def make_mnist_transforms(image_set, cutout):
    mnist_mean = [0.5]
    mnist_std = [0.5]

    if image_set == 'train':
        trans = [
            transforms.RandomCrop(28, padding=2),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mnist_mean, mnist_std)
        ]
        if cutout:
            trans.append(cutout_transform)
        return transforms.Compose(trans)

    if image_set == 'val':
        trans = [
            transforms.ToTensor(),
            transforms.Normalize(mnist_mean, mnist_std)
        ]
        return transforms.Compose(trans)

    raise ValueError(f'unknown {image_set}')


def build_mnist_dataset(with_cutout=False):
    train_dataset = nni.trace(MNIST)(root=DATA_STORAGE_PATH, train=True, download=True,
                                     transform=make_mnist_transforms('train', cutout=with_cutout))
    valid_dataset = nni.trace(MNIST)(root=DATA_STORAGE_PATH, train=False, download=True,
                                     transform=make_mnist_transforms('val', cutout=with_cutout))

    return train_dataset, valid_dataset


def build_fashion_mnist_dataset(with_cutout=False):
    train_dataset = nni.trace(FashionMNIST)(root=DATA_STORAGE_PATH, train=True, download=True,
                                            transform=make_mnist_transforms('train', cutout=with_cutout))
    valid_dataset = nni.trace(FashionMNIST)(root=DATA_STORAGE_PATH, train=False, download=True,
                                            transform=make_mnist_transforms('val', cutout=with_cutout))

    return train_dataset, valid_dataset
