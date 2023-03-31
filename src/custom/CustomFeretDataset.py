import torch
from torch.utils.data import Dataset
from PIL import Image
from facenet_pytorch import MTCNN
import torchvision.transforms as T

class CustomFeretDataset(Dataset):
    def __init__(self, data_dir, images_names, classes, transform=None, mtcnn_detect=False):
        self.data_dir = data_dir
        self.classes = classes
        self.images_names = images_names
        self.images_paths = self.get_images_paths()
        self.labels = self.get_labels()
        self.transform = transform
        self.mtcnn_detect = mtcnn_detect
        self.mtcnn = MTCNN() if self.mtcnn_detect else None

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = Image.open(self.get_image_path(idx))

        if self.mtcnn_detect:
            save_image = image.copy()
            image = self.mtcnn(image)
            if image is None:  # handle no face detect
                image = save_image.copy()
            else:
                image = image / 2 + 0.5  # fix clipping

        # image = image.type(torch.FloatTensor)
        if self.transform:
            image = self.transform(image)

        return image, self.classes.index(self.labels[idx])

    def get_image_path(self, idx):
        return self.images_paths[idx]

    def get_labels(self):
        return [path.split('_')[1].split('/')[-1] for path in self.images_paths]

    def get_images_paths(self):
        return [self.data_dir + '/' + name.split('_')[0] + '/' + name + '.jpg' for name in self.images_names]
