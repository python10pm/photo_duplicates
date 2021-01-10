import os
import sys

import numpy as np
import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

class PhotoToArray(object):

    def __init__(self, path_input, path_output, height, width, verbose = False):
        '''
        Constructor for the class.
        '''
        self.path_input = path_input
        self.path_output = path_output
        self.height = height
        self.width = width
        self.verbose = verbose

    def get_full_path_photos(self):
        '''
        Creates a list of full path for each photo in the original folder.
        '''
        return [os.path.join(self.path_input, file) for file in os.listdir(self.path_input)]

    def check_output_folder(self):
        '''
        Checks if the output folder exists and creates it if needed.
        '''
        if not os.path.isdir(self.path_output):
            os.mkdir(self.path_output)

    def photo_resize(self):
        '''
        Iterates over all original photos and reshapes them into the specified height x width.
        '''
        photos_path = self.get_full_path_photos()

        self.check_output_folder()

        for photo_path in photos_path:
            if ".DS_Store" not in photo_path:

                img = Image.open(photo_path)
                size = img.size
                if self.verbose: print(f"Size of {photo_path} is {size}")

                img = img.resize((self.height, self.width), resample = Image.BILINEAR)
                img.save(os.path.join(self.path_output, os.path.basename(photo_path)))

    def photo_to_numeric(self):
        '''
        Iterates over all reshaped photos and loads them as a numpy array.
        Returns a list of numpy arrays.
        '''
        photo_names = []
        photo_arrays = []

        for photo in os.listdir(self.path_output):
            photo_path = os.path.join(self.path_output, photo)

            photo_array = Image.open(photo_path)
            photo_array = np.asarray(photo_array).ravel()

            photo_names.append(photo)
            photo_arrays.append(photo_array)

            if self.verbose: print(photo_path, photo_array.shape)

        return photo_names, photo_arrays

    def create_photo_df(self):
        '''
        Creates a standard DataFrame from the numeric photo data
        '''
        photo_names, photo_arrays = self.photo_to_numeric()
        column_names = [f"pixel_{i}" for i in range(photo_arrays[0].shape[0])]
        photo_df = pd.DataFrame(data = photo_arrays, index = photo_names, columns = column_names)

        return photo_df

def show_similar_photos(photo_name, photo_similarity_df, save = True):
    '''
    Shows the 2 most similar photos.
    '''
    assert photo_name in photo_similarity_df.columns, "The selected photo is not found in the df"

    photo_names = photo_similarity_df[photo_name].sort_values(ascending = False)[:2].index.tolist()
    photo_score = photo_similarity_df[photo_name].sort_values(ascending = False)[:2].values

    photo_1_name = photo_names[0]
    photo_2_name = photo_names[1]

    fig = plt.figure(figsize = (10, 10))
    ax1, ax2 = fig.subplots(1, 2)
    
    ax1.imshow(Image.open(os.path.join(path_output, photo_1_name)))
    ax2.imshow(Image.open(os.path.join(path_output, photo_2_name)))

    fig.suptitle("Cosime similarity of {}".format(round(photo_score[1], 5)))

    plt.savefig("similar_photos.png") if save else plt.show()

if __name__ == "__main__":

    print("Running script as __main__")

    # create input and output folder and other variables
    path_input = os.path.join(os.getcwd(), "photos")
    path_output = os.path.join(os.getcwd(), "photos_output")

    height = 256
    width = 256

    # instanciates the class and runs the resize and numeric conversion of photos
    photo_to_array = PhotoToArray(
        path_input = path_input, 
        path_output = path_output,
        height = height,
        width = width
        )

    photo_to_array.photo_resize()
    photo_df = photo_to_array.create_photo_df()

    # computes the cosine similarity between photos
    scaler = MinMaxScaler()
    photo_df_ = scaler.fit_transform(photo_df)
    
    # photo_df_ = photo_df.copy(deep = True)
    # photo_df_ = photo_df_/255
    
    photo_similarity = cosine_similarity(photo_df_)

    photo_similarity_df = pd.DataFrame(photo_similarity, index = photo_df.index, columns = photo_df.index)
    print(photo_similarity_df.head())

    # creates a png image with 2 photos compared and their cosine similarity score
    photo_to_check = "IMG_20201220_141433.jpg"
    show_similar_photos(
        photo_name = photo_to_check, 
        photo_similarity_df = photo_similarity_df
        )