import os
import nni
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import CIFAR100
from engine.dataset.transformers import cutout_transform
from common.config import DATA_STORAGE_PATH


class CIFAR10Folder(Dataset):
    def __init__(self, path, transform=None):
        self.path = path
        self.transform = transform
        self.classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                        'dog', 'frog', 'horse', 'ship', 'truck']
        self.classname_to_id = {classname: idx for idx, classname in enumerate(self.classes)}
        self.images = self.__load_images()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_path = self.images[index]
        image = Image.open(image_path)

        if self.transform is not None:
            image = self.transform(image)

        *prefix, classname, filename = image_path.split(os.sep)
        label = self.classname_to_id[classname]

        return image, label

    def __load_images(self):
        image_paths = []
        for class_name in os.listdir(self.path):
            class_path = os.path.join(self.path, class_name)
            image_names = os.listdir(class_path)
            image_paths.extend([os.path.join(
                class_path, im_name) for im_name in image_names])
        return image_paths


def make_cifar_transforms(image_set, cutout):
    cifar_mean = [0.49139968, 0.48215827, 0.44653124]
    cifar_std = [0.24703233, 0.24348505, 0.26158768]

    if image_set == 'train':
        trans = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(cifar_mean, cifar_std)
        ]
        if cutout:
            trans.append(cutout_transform)
        return transforms.Compose(trans)

    if image_set == 'val':
        trans = [
            transforms.ToTensor(),
            transforms.Normalize(cifar_mean, cifar_std)
        ]
        return transforms.Compose(trans)

    raise ValueError(f'unknown {image_set}')


def build_cifar10_dataset(with_cutout=False):
    train_dataset = nni.trace(CIFAR10Folder)(os.path.join(DATA_STORAGE_PATH, 'cifar10', 'train'),
                                             transform=make_cifar_transforms('train', with_cutout))
    valid_dataset = nni.trace(CIFAR10Folder)(os.path.join(DATA_STORAGE_PATH, 'cifar10', 'test'),
                                             transform=make_cifar_transforms('val', with_cutout))
    return train_dataset, valid_dataset


def build_cifar100_dataset(with_cutout=False):
    train_dataset = nni.trace(CIFAR100)(root=DATA_STORAGE_PATH, train=True, download=True,
                                        transform=make_cifar_transforms('train', cutout=with_cutout))
    valid_dataset = nni.trace(CIFAR100)(root=DATA_STORAGE_PATH, train=False, download=True,
                                        transform=make_cifar_transforms('val', cutout=with_cutout))

    return train_dataset, valid_dataset
