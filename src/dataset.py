from torch.utils.data import DataLoader
from src.custom.CustomFeretDataset import CustomFeretDataset
import random
import matplotlib.pyplot as plt
import glob
import numpy as np
from torchvision import transforms


def get_dataset(DATASET, train_images_name, test_images_name):
    """
    Get dataset using a custom dataset

    :param DATASET: dataset name
    :param train_images_name: dataset train images name
    :param test_images_name: dataset test images name
    :return: dataset object
    """
    dataset_train = CustomFeretDataset(
        DATASET['images_dir'],
        images_names=train_images_name,
        mtcnn_detect=DATASET['mtcnn_detect'],
        transform=transforms.Compose({
            transforms.Resize(DATASET['size']),
        })
    )

    dataset_test = CustomFeretDataset(
        DATASET['images_dir'],
        images_names=test_images_name,
        mtcnn_detect=DATASET['mtcnn_detect'],
        transform=transforms.Compose({
            transforms.Resize(DATASET['size']),
        })
    )

    iterate_dataset(dataset_train)
    iterate_dataset(dataset_test)

    return dataset_train, dataset_test


def get_dataset_loader(dataset, DATASET):
    """
    Get dataset loader

    :param dataset: dataset
    :param DATASET: info
    """

    dataset_loader = DataLoader(
        dataset,
        batch_size=DATASET['data_loader']['batch_size'],
        shuffle=DATASET['data_loader']['shuffle'],
        num_workers=DATASET['data_loader']['num_workers']
    )

    check_dataset_loader(dataset_loader)

    return dataset_loader


def iterate_dataset(dataset):
    """
    Iterate through dataset and plot random images

    :param: dataset
    """
    for i in random.sample(range(0, len(dataset)), 3):
        sample, label = dataset[i]

        plt.figure()
        plt.imshow(sample)
        plt.title('Subject: ' + label)
        plt.show()


def check_dataset_loader(dataset_loader):
    """
    Check dataset_loader

    :param dataset_loader: dataset_loader
    """

    for i in range(0, 10):
        frames, labels = next(iter(dataset_loader))
        print(f"Feature batch shape: {frames.size()}")
        print(f"Labels batch shape: {np.shape(labels)}")
        img = frames[0].squeeze()
        label = labels[0]
        plt.imshow(img, cmap="gray")
        plt.title('Subject' + label)
        plt.show()


def get_dataset_images_name(DATASET, split=True):
    paths = glob.glob(DATASET['images_dir'] + '/*/*.jpg')

    names = [path.split('\\')[-1].replace('.jpg', '') for path in paths]

    if not split:
        return names

    test_images_name = []
    for test_file in DATASET['test_files']:
        file = open(test_file, 'r')
        for line in file.readlines():
            test_images_name.append(line.split()[1].replace('.ppm', ''))
        file.close()

    test_images_name = set(test_images_name)
    train_images_name = set(names) - test_images_name

    return train_images_name, test_images_name
