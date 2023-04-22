import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import torchvision
from skimage.feature import hog, local_binary_pattern
import numpy as np
import src.utils as utils

class CustomFeretDataset(Dataset):
    def __init__(self, data_dir, images_names, classes, transform=None, use_cache=False, feature_extraction=None, augmentation=False):
        self.data_dir = data_dir
        self.classes = classes
        self.images_names = images_names
        self.images_paths = self.get_images_paths()
        self.labels = self.get_labels()
        self.transform = transform
        self.cached_data = dict()
        self.cached_data_features = dict()
        self.use_cache = use_cache
        self.feature_extraction = feature_extraction
        self.augmentation = augmentation
        self.augmentation_transform = torchvision.transforms.Compose([
            torchvision.transforms.ColorJitter(hue=0.05, saturation=0.05, contrast=0.1, brightness=0.1),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomAdjustSharpness(sharpness_factor=3, p=0.5),
            torchvision.transforms.RandomRotation(degrees=10),
        ])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if not self.use_cache or not self.get_image_path(idx) in self.cached_data:
            image = Image.open(self.get_image_path(idx))
            convert_tensor = transforms.ToTensor()
            image = convert_tensor(image)

            if self.transform:
                image = self.transform(image)

            features = torch.tensor([])
            if self.feature_extraction:
                if self.feature_extraction == 'LBP':
                    lbp = local_binary_pattern(utils.rgb2gray(image.permute(1, 2, 0).numpy()), P=16, R=3*16, method='uniform')
                    features, bins = utils.create_histograms(lbp, sub_images_num=3, bins_per_sub_images=16)
                    features = torch.tensor(features, dtype=torch.float32)
                    # print(features)
                    #
                    # # print(np.max(lbp.ravel()), features)
                    # # print(np.min(features), np.max(features), np.max(lbp))
                    # # print(np.shape(features))
                    # # print(np.shape(features), np.shape(bins))
                    # # print('da')
                    #
                    # fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5), dpi=300)
                    # axes[0].imshow(image.permute(1, 2, 0).numpy(), cmap="gray")
                    # axes[1].hist(range(0, 16*9), 16*9, weights=features)
                    # fig.tight_layout()
                    # plt.savefig('data/results/dataset/lbp/lbp_sample_' + str(self.get_image_path(idx)).split('/')[-1], dpi=fig.dpi)

                if self.feature_extraction == 'HOG':
                    # image = utils.rgb2gray(image.permute(1, 2, 0).numpy())
                    # features = np.reshape(image, (np.shape(image)[0] * np.shape(image)[1]))

                    features = hog(image.permute(1, 2, 0).numpy(), orientations=8, pixels_per_cell=(8, 8),
                                              cells_per_block=(2, 2), visualize=False, channel_axis=-1, block_norm='L2-Hys', transform_sqrt=True)
                    # print(np.min(features), np.max(features))
                    # print(np.shape(features))
                    #
                    # fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5), dpi=300)
                    # axes[0].imshow(image.permute(1, 2, 0).numpy(), cmap="gray")
                    # axes[1].imshow(hog_image, cmap="gray")
                    # fig.tight_layout()
                    # plt.savefig('data/results/dataset/hog/hog_sample_' + str(self.get_image_path(idx)).split('/')[-1], dpi=fig.dpi)

            if self.use_cache:
                self.cached_data[self.get_image_path(idx)] = image

                if self.feature_extraction:
                    self.cached_data_features[str(self.get_image_path(idx)) + '_features'] = features
        else:
            image = self.cached_data[self.get_image_path(idx)]

            if self.augmentation:
                # image_copy = image.permute(1, 2, 0).numpy().copy()
                image = self.augmentation_transform(image)

                # # example augmentation
                # fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5), dpi=300)
                # axes[0].imshow(image_copy, cmap="gray")
                # axes[1].imshow(image.permute(1, 2, 0).numpy(), cmap="gray")
                # fig.tight_layout()
                # plt.savefig('data/results/dataset/augmentation/sample_' + str(self.get_image_path(idx)).split('/')[-1], dpi=fig.dpi)

            features = torch.tensor([])
            if self.feature_extraction:
                features = self.cached_data_features[str(self.get_image_path(idx)) + '_features']

        return image, self.classes.index(self.labels[idx]), features

    def get_image_path(self, idx):
        return self.images_paths[idx]

    def get_labels(self):
        return [path.split('_')[1].split('/')[-1] for path in self.images_paths]

    def get_images_paths(self):
        return [self.data_dir + '/' + name.split('_')[0] + '/' + name + '.jpg' for name in self.images_names]
