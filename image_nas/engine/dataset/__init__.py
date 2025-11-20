from engine.dataset.cifar import build_cifar10_dataset, build_cifar100_dataset
from engine.dataset.mnist import build_mnist_dataset, build_fashion_mnist_dataset
from engine.dataset.image_folder import build_image_folder_dataset


def build_dataset(dataset_name: str = 'cifar10', with_cutout=False, normalization_online=False):
    if dataset_name == 'cifar10':
        return build_cifar10_dataset(with_cutout)
    if dataset_name == 'cifar100':
        return build_cifar100_dataset(with_cutout)
    elif dataset_name == 'mnist':
        return build_mnist_dataset(with_cutout)
    elif dataset_name == 'fashion-mnist':
        return build_fashion_mnist_dataset(with_cutout)
    else:
        return build_image_folder_dataset(dataset_name,
                                          with_cutout=with_cutout,
                                          normalization_online=normalization_online)


if __name__ == '__main__':
    dataset_name = 'mnist'
    train_dataset, valid_dataset = build_dataset(dataset_name, normalization_online=False)
    print(train_dataset.classes)
    print(valid_dataset.classes)
