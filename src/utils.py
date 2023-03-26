import os
import glob
import bz2, shutil
from PIL import Image

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