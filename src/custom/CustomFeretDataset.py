import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


class CustomFeretDataset(Dataset):
    def __init__(self, data_dir, images_names, classes, transform=None, use_cache=False):
        self.data_dir = data_dir
        self.classes = classes
        self.images_names = images_names
        self.images_paths = self.get_images_paths()
        self.labels = self.get_labels()
        self.transform = transform
        self.cached_data = dict()
        self.use_cache = use_cache

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if not self.use_cache or not self.get_image_path(idx) in self.cached_data:
            image = Image.open(self.get_image_path(idx))
            convert_tensor = transforms.ToTensor()
            image = convert_tensor(image)

            if self.transform:
                image = self.transform(image)

            if self.use_cache:
                self.cached_data[self.get_image_path(idx)] = image
        else:
            image = self.cached_data[self.get_image_path(idx)]

        return image, self.classes.index(self.labels[idx])

    def get_image_path(self, idx):
        return self.images_paths[idx]

    def get_labels(self):
        return [path.split('_')[1].split('/')[-1] for path in self.images_paths]

    def get_images_paths(self):
        return [self.data_dir + '/' + name.split('_')[0] + '/' + name + '.jpg' for name in self.images_names]