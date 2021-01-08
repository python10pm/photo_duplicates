import os
import sys

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

class PhotoToArray(object):

    def __init__(self, path_input, path_output, height = 252, width = 252, verbose = False):
        self.path_input = path_input
        self.path_output = path_output
        self.height = height
        self.width = width
        self.verbose = verbose

    def _get_full_path_photos(self):
        return [os.path.join(self.path_input, file) for file in os.listdir(self.path_input)]

    def _check_output_folder(self):
        if not os.path.isdir(self.path_output):
            os.mkdir(self.path_output)

    def photo_resize(self):

        photos_path = self._get_full_path_photos()

        self._check_output_folder()

        for photo_path in photos_path:
            if ".DS_Store" not in photo_path:

                img = Image.open(photo_path)
                size = img.size
                if self.verbose: print(f"Size of {photo_path} is {size}")

                img = img.resize((self.height, self.width), resample = Image.BILINEAR)
                img.save(os.path.join(self.path_output, os.path.basename(photo_path)))

    def photo_to_numeric(self):
        photo_arrays = []
        for photo in os.listdir(self.path_output):
            photo_array = Image.open(photo).asarray().ravel()
            photo_arrays.append(photo_array)

        return photo_arrays

# https://www.geeksforgeeks.org/how-to-convert-images-to-numpy-array/

if __name__ == "__main__":
    print("Running script as __main__")
    path_input = os.path.join(os.getcwd(), "photos")
    path_output = os.path.join(os.getcwd(), "photos_output")
    photo_to_array = PhotoToArray(path_input = path_input, path_output = path_output)
    print(photo_to_array.photo_resize())
    print(len(photo_to_array.photo_to_numeric()))