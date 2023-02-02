import os
import numpy as np
from tqdm import tqdm
from skimage.io import imread
from skimage.transform import resize
    
def load_data(dataset="UNet_dataScienceBowl", transformation = None, n_train = None, n_test = None):
    """
    Load images as numpy arrays from raw images and combine multiple masks into one mask

    Args:
        dataset (str, optional): Name of dataset. Defaults to "UNet_dataScienceBowl".
        transformation (_type_, optional): Data transformation. Defaults to None.
        n_train (_type_, optional): Number of training data. Defaults to None.
        n_test (_type_, optional): Number of test data. Defaults to None.

    Returns:
        numpy arrays: data for training and testing
    """

    np.random.seed(0)

    # Define the width, height, number of channels of input images
    IMG_WIDTH = 256
    IMG_HEIGHT = 256
    IMG_CHANNELS = 3

    # Specify path to dataset 
    DATA_PATH = 'data/'
    TRAIN_PATH = DATA_PATH + 'stage1_train/'  
    TEST_PATH = DATA_PATH + 'stage1_test/'  

    # next(os.walk(TRAIN_PATH)) -- [[PATH], [ids], []]
    # Get all image ids
    train_ids = next(os.walk(TRAIN_PATH))[1]
    test_ids = next(os.walk(TEST_PATH))[1]
    
    X_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
    Y_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, 1), dtype=bool)
    
    # Ref: # https://www.kaggle.com/code/mbadrismail/unet-datasciencebowl
    # Converting images as np array and combine masks as one mask
    for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
        path = TRAIN_PATH + id_

        # Constraint number of channels
        img = imread(path + '/images/' + id_ + '.png')[:, :, :IMG_CHANNELS]
        img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
        X_train[n] = img  
        
        mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=bool)

        # [[PATH], [], [ids]]
        # Go through all masks and collect its segments into one mask
        for mask_file in next(os.walk(path + '/masks/'))[2]:  
            mask_ = imread(path + '/masks/' + mask_file)
            mask_ = np.expand_dims(resize(mask_, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True), axis=-1)
            mask = np.maximum(mask, mask_)

        Y_train[n] = mask

    X_test = np.zeros((len(test_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
    for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):
        path = TEST_PATH + id_
        img = imread(path + '/images/' + id_ + '.png')[:, :, :IMG_CHANNELS]
        img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
        X_test[n] = img
        
    return X_train, Y_train, X_test

if __name__ == "__main__":
    X_train, Y_train, X_test = load_data()
        
    np.save("./data/X_train.npy", X_train)
    np.save("./data/Y_train.npy", Y_train)
    np.save("./data/X_test.npy", X_test)
    