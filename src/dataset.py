from torch.utils.data import DataLoader
from src.custom.CustomFeretDataset import CustomFeretDataset
import random
import matplotlib.pyplot as plt
import glob
import numpy as np
from torchvision import transforms
import xmltodict
import collections
import src.utils as utils


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
    :return: dataset loader object
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
    """
    Get all images name for dataset train & test

    :param DATASET: DATASET
    :param split: split
    :return: dataset images names
    """

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


def plot_dataset_visualisation(DATASET):
    """
    Plot all info for dataset

    :param DATASET: DATASET
    """
    subjects_info = get_subjects_information(DATASET)
    recordings_info = get_recordings_info(DATASET)

    dataset_distribution(DATASET)
    dataset_others_distribution(DATASET, subjects_info, recordings_info)


def get_subjects_information(DATASET):
    """
    Get subjects information from XML file converted to dictionary

    :param DATASET: DATASET
    :return: subjects information dict
    """
    with open(DATASET['subjects_info'], 'r', encoding='utf-8') as file:
        xml_subjects = file.read()

    return xmltodict.parse(xml_subjects)


def get_recordings_info(DATASET):
    """
    Get subjects information from XML file converted to dictionary

    :param DATASET: DATASET
    :return: subjects information dict
    """
    with open(DATASET['recordings_info'], 'r', encoding='utf-8') as file:
        xml_recordings_info = file.read()

    return xmltodict.parse(xml_recordings_info)


def dataset_distribution(DATASET):
    """
    Plot dataset classes distribution

    :param DATASET: DATASET
    """
    paths_dir_subjects = glob.glob(DATASET['images_dir'] + '/*')
    classes_idx = [int(idx.split('\\')[-1]) for idx in paths_dir_subjects]

    dataset_classes = []
    dataset_distr = []
    x_ticks = []
    x_labels = []
    for i, dir in enumerate(paths_dir_subjects):
        if int(dir.split('\\')[-1]) in classes_idx:
            dataset_distr.append(len(glob.glob(dir + '/*.jpg')))
            dataset_classes.append(dir.split('\\')[-1])
        else:
            dataset_distr.append(0)
            dataset_classes.append(int(dataset_classes[-1]) + 1)

        if i % 20 == 0:
            x_ticks.append(i)
            x_labels.append(dir.split('\\')[-1])

    fig = plt.figure(figsize=(30, 10), dpi=900)
    plt.bar(dataset_classes, dataset_distr, color='maroon')
    plt.xlabel("Dataset class (subject ID)", fontsize=12)
    plt.ylabel("No. of images", fontsize=12)
    plt.title('Dataset distribution' + ' | ' + DATASET['name'], fontsize=18)
    plt.margins(0)
    ax = plt.gca()
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels, rotation=30, fontsize=8, horizontalalignment='right')
    plt.tight_layout()
    plt.savefig('data/results/dataset_distribution.jpg', dpi=fig.dpi)
    # plt.show()

    print('Min:', dataset_classes[np.argmin(dataset_distr)], dataset_distr[np.argmin(dataset_distr)])
    print('Max:', dataset_classes[np.argmax(dataset_distr)], dataset_distr[np.argmax(dataset_distr)])


def dataset_others_distribution(DATASET, subjects_info, recordings_info):
    """
    Plot dataset classes distribution by gender

    :param DATASET: DATASET
    :param subjects_info: subjects information
    :param recordings_info: subjects recordings information
    """

    info = {
        'gender': [],
        'race': [],
        'YOB': [],
        'glasses': [],
        'beard': [],
        'mustache': [],
        'pose': [],
    }
    for subject_info in subjects_info['Subjects']['Subject']:
        info['gender'].append(subject_info['Gender']['@value'])
        info['race'].append(subject_info['Race']['@value'])
        info['YOB'].append(subject_info['YOB']['@value'])

    for subject_info_recordings in recordings_info['Recordings']['Recording']:
        info['glasses'].append(subject_info_recordings['Subject']['Application']['Face']['Wearing']['@glasses'])
        info['beard'].append(subject_info_recordings['Subject']['Application']['Face']['Hair']['@beard'])
        info['mustache'].append(subject_info_recordings['Subject']['Application']['Face']['Hair']['@mustache'])
        info['pose'].append(subject_info_recordings['Subject']['Application']['Face']['Pose']['@name'])

    # gender
    counter = collections.Counter(info['gender'])
    distr = []
    data = []

    for key, count in sorted(counter.items()):
        data.append(key)
        distr.append(count)

    distr = distr / np.sum(distr) * 100

    fig = plt.figure(figsize=(10, 10), dpi=900)
    plt.bar(data, distr, color='darkgreen', width=0.5)
    plt.xlabel("Gender", fontsize=14)
    plt.ylabel("Subjects percent [%]", fontsize=14)
    plt.title('Dataset gender distribution' + ' | ' + DATASET['name'], fontsize=20)
    ax = plt.gca()
    ax.tick_params(axis='both', labelsize=14)
    plt.tight_layout()
    utils.add_labels(data, distr)
    plt.savefig('data/results/dataset_gender_distribution.jpg', dpi=fig.dpi)

    # race
    counter = collections.Counter(info['race'])
    distr = []
    data = []
    for key, count in sorted(counter.items()):
        data.append(key)
        distr.append(count)

    distr = distr / np.sum(distr) * 100

    fig = plt.figure(figsize=(10, 10), dpi=900)
    plt.bar(data, distr, color='darkorange', width=0.5)
    plt.xlabel("Race", fontsize=14)
    plt.ylabel("Subjects percent [%]", fontsize=14)
    plt.title('Dataset race distribution' + ' | ' + DATASET['name'], fontsize=20)
    ax = plt.gca()
    ax.set_xticks(data)
    ax.set_xticklabels(data, rotation=45, ha='right', rotation_mode='anchor')
    ax.tick_params(axis='y', labelsize=14)
    plt.tight_layout()
    utils.add_labels(data, distr)
    plt.savefig('data/results/dataset_race_distribution.jpg', dpi=fig.dpi)

    # YOB
    counter = collections.Counter(info['YOB'])
    distr = []
    data = []
    for key, count in sorted(counter.items()):
        data.append(key)
        distr.append(count)

    distr = distr / np.sum(distr) * 100

    fig = plt.figure(figsize=(10, 10), dpi=900)
    plt.bar(data, distr, color='dodgerblue', width=0.5)
    plt.xlabel("Year of born", fontsize=14)
    plt.ylabel("Subjects percent [%]", fontsize=14)
    plt.title('Dataset YOB distribution' + ' | ' + DATASET['name'], fontsize=20)
    ax = plt.gca()
    ax.set_xticks(data)
    ax.set_xticklabels(data, rotation=45, ha='right', rotation_mode='anchor')
    ax.tick_params(axis='y', labelsize=14)
    plt.tight_layout()
    utils.add_labels(data, distr)
    plt.savefig('data/results/dataset_yob_distribution.jpg', dpi=fig.dpi)

    # mustache
    counter = collections.Counter(info['mustache'])
    distr = []
    data = []
    for key, count in sorted(counter.items()):
        data.append(key)
        distr.append(count)

    distr = distr / np.sum(distr) * 100

    fig = plt.figure(figsize=(10, 10), dpi=900)
    plt.bar(data, distr, color='brown', width=0.5)
    plt.xlabel("Mustache", fontsize=14)
    plt.ylabel("Images percent [%]", fontsize=14)
    plt.title('Dataset mustache distribution' + ' | ' + DATASET['name'], fontsize=20)
    ax = plt.gca()
    ax.set_xticks(data)
    ax.tick_params(axis='both', labelsize=14)
    plt.tight_layout()
    utils.add_labels(data, distr)
    plt.savefig('data/results/dataset_mustache_distribution.jpg', dpi=fig.dpi)

    # glasses
    counter = collections.Counter(info['glasses'])
    distr = []
    data = []
    for key, count in sorted(counter.items()):
        data.append(key)
        distr.append(count)

    distr = distr / np.sum(distr) * 100

    fig = plt.figure(figsize=(10, 10), dpi=900)
    plt.bar(data, distr, color='royalblue', width=0.5)
    plt.xlabel("Glasses", fontsize=14)
    plt.ylabel("Images percent [%]", fontsize=14)
    plt.title('Dataset glasses distribution' + ' | ' + DATASET['name'], fontsize=20)
    ax = plt.gca()
    ax.set_xticks(data)
    ax.tick_params(axis='both', labelsize=14)
    plt.tight_layout()
    utils.add_labels(data, distr)
    plt.savefig('data/results/dataset_glasses_distribution.jpg', dpi=fig.dpi)

    # beard
    counter = collections.Counter(info['beard'])
    distr = []
    data = []
    for key, count in sorted(counter.items()):
        data.append(key)
        distr.append(count)

    distr = distr / np.sum(distr) * 100

    fig = plt.figure(figsize=(10, 10), dpi=900)
    plt.bar(data, distr, color='indianred', width=0.5)
    plt.xlabel("Beard", fontsize=14)
    plt.ylabel("Images percent [%]", fontsize=14)
    plt.title('Dataset beard distribution' + ' | ' + DATASET['name'], fontsize=20)
    ax = plt.gca()
    ax.set_xticks(data)
    ax.tick_params(axis='both', labelsize=14)
    plt.tight_layout()
    utils.add_labels(data, distr)
    plt.savefig('data/results/dataset_beard_distribution.jpg', dpi=fig.dpi)

    # pose
    counter = collections.Counter(info['pose'])
    distr = []
    data = []
    for key, count in sorted(counter.items()):
        if key == 'fb':  # fa & fb same 0 grades
            distr[0] = distr[0] + count
        else:
            data.append(key)
            distr.append(count)

    data_labels = {
        'fa': '0°',
        'fb': '0°',
        'hl': '+67.5°',
        'hr': '-67.5°',
        'pl': '+90°',
        'pr': '-90°',
        'ql': '+22.5°',
        'qr': '-22.5°',
        'ra': '+45°',
        'rb': '+15°',
        'rc': '-15°',
        'rd': '-45°',
        're': '-75°',
    }

    data = [data_labels[d] for d in data]
    distr = distr / np.sum(distr) * 100

    fig = plt.figure(figsize=(10, 10), dpi=900)
    plt.bar(data, distr, color='darkviolet', width=0.5)
    plt.xlabel("Poses", fontsize=14)
    plt.ylabel("Images percent [%]", fontsize=14)
    plt.title('Dataset poses distribution' + ' | ' + DATASET['name'], fontsize=20)
    ax = plt.gca()
    ax.set_xticks(data)
    ax.tick_params(axis='both', labelsize=14)
    plt.tight_layout()
    utils.add_labels(data, distr)
    plt.savefig('data/results/dataset_poses_distribution.jpg', dpi=fig.dpi)
