import os
import glob
import bz2, shutil
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


def decompress_bz2_dataset():
    # this path must exist
    compressed_files_path = "data/colorferet/dvd2/data/images"

    destination_folder = "data/colorferet/converted_images/images"

    # get list with filenames (strings)
    dirListing = glob.glob(compressed_files_path + '/*/*')

    for file in dirListing:
        # ^ this is only filename.ext
        if ".bz2" in file:
            # concatenation of directory path and filename.bz2
            existing_file_path = file

            split = file.split('\\')
            new_file_path = destination_folder + '/' + split[-2] + '/' + split[-1]

            if not os.path.exists(destination_folder + '/' + split[-2]):
                os.makedirs(destination_folder + '/' + split[-2], exist_ok=True)

            with bz2.BZ2File(existing_file_path) as fr, open(new_file_path[:-4], "wb") as fw:
                shutil.copyfileobj(fr, fw)


def convert_to_jpg_dataset():
    src = 'data/colorferet/converted_images/images'
    dst = 'data/colorferet/converted_images/images'
    for root, dirs, filenames in os.walk(src, topdown=False):
        for filename in filenames:
            if ".ppm" in filename:
                inputfile = os.path.join(root, filename)
                outputfile = os.path.join(dst, filename.split('_')[0] + '/' + filename.replace(".ppm", ".jpg"))

                im = Image.open(inputfile)
                im.save(outputfile)

                os.remove(inputfile)


def add_labels(x, y):
    for i in range(len(x)):
        plt.text(i, y[i] + 0.2, "{:.2f}".format(y[i]) + ' %', ha='center', fontsize=16)


def rgb2gray(rgb):
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray


def create_histograms(image, sub_images_num, bins_per_sub_images):
    grid = np.arange(0, image.shape[1] + 1, image.shape[1] // sub_images_num)

    sub_image_histograms = []
    sub_image_bins = []

    for i in range(1, len(grid)):
        for j in range(1, len(grid)):
            sub_image = image[grid[i - 1]:grid[i], grid[j - 1]:grid[j]]
            sub_image_histogram, bins = np.histogram(sub_image, bins=bins_per_sub_images, density=True)

            sub_image_histograms.extend(sub_image_histogram)
            sub_image_bins.append(sub_image_bins)

    return sub_image_histograms, sub_image_bins
