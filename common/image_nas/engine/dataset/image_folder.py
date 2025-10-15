import os
import nni
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import transforms
from engine.dataset import util
from engine.dataset.transformers import cutout_transform
from common.config import DATA_STORAGE_PATH


def make_image_folder_transforms(image_set,
                                 mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225],
                                 cutout=False):
    if image_set == 'train':
        trans = [
            transforms.RandomResizedCrop(224),  # 32 --> 224
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ]
        if cutout:
            trans.append(cutout_transform)
        return transforms.Compose(trans)
    elif image_set == 'val':
        trans = [
            transforms.Resize(256),  # 38 --> 256
            transforms.CenterCrop(224),  # 32 --> 224
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ]
        return transforms.Compose(trans)

    raise ValueError(f'unknown {image_set}')


def calculate_dataset_mean_std(dataset_path):
    trans = transforms.Compose([
        transforms.Resize(38),
        transforms.CenterCrop(32),
        transforms.ToTensor()
    ])
    image_dataset = ImageFolder(root=os.path.join(dataset_path, 'train'), transform=trans)
    data_loader = DataLoader(image_dataset, batch_size=64, num_workers=0)
    mean, std = util.batch_mean_and_std(data_loader)
    return mean, std


def build_image_folder_dataset(dataset_name, with_cutout=False, normalization_online=False):
    dataset_path = os.path.join(DATA_STORAGE_PATH, dataset_name)

    if not os.path.exists(dataset_path):
        raise ValueError("There is not such dataset `{}`.".format(dataset_name))

    if normalization_online:
        mean, std = calculate_dataset_mean_std(dataset_path)
    else:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

    train_dataset = nni.trace(ImageFolder)(root=os.path.join(dataset_path, 'train'),
                                           transform=make_image_folder_transforms('train',
                                                                                  mean=mean,
                                                                                  std=std,
                                                                                  cutout=with_cutout))
    valid_dataset = nni.trace(ImageFolder)(root=os.path.join(dataset_path, 'val'),
                                           transform=make_image_folder_transforms('val',
                                                                                  mean=mean,
                                                                                  std=std,
                                                                                  cutout=with_cutout))
    return train_dataset, valid_dataset
