"""
    Jacob Kaplan

    descriptor.py

    Takes in a directory of sub-directories of images and produces a descriptor
    file for all the images found in the sub-directories.

                                ,:'/   _..._
                                // ( `""-.._.'
                                \| /    6\___
                                |     6      4
                                |            /
                                \_       .--'
                                (_'---'`)
                                / `'---`()
                              ,'        |
              ,            .'`          |
              )\       _.-'             ;
             / |    .'`   _            /
           /` /   .'       '.        , |
          /  /   /           \   ;   | |
          |  \  |            |  .|   | |
           \  `"|           /.-' |   | |
            '-..-\       _.;.._  |   |.;-.
                  \    <`.._  )) |  .;-. ))
                  (__.  `  ))-'  \_    ))'
                      `'--"`       `""`
        I'll describe each image for you...okay that first one is gray...the
        second one is also gray...the next one's gray...
"""

import os
import sys
import pickle
import cv2 as cv
import numpy as np

def read_train_dir(train_dir):
    """
    Take in a directory of sub-directories of images
    Go through each sub-directory that matches one of the backgrounds
        (grass, ocean, readcarpet, road, wheatfield)
    Get all the images in each sub-directory and put it into a list
    Append that list of all images in sub-directory to train_dir_imgs list
    Return train_dir_imgs list which will contain all the images in all the
        sub-directories sorted by background classification
    """
    train_dir_imgs = list()
    img_dirs = ["grass", "ocean", "redcarpet", "road", "wheatfield"]
    for fil in os.listdir(train_dir):
        if fil in img_dirs:
            images = find_images(train_dir + "/" + fil)
            train_dir_imgs.append(images)
    return train_dir_imgs

def find_images(dir):
    """
    Take in directory of images, open directory, read in each image,
    add to list of images, return list of images
    """
    images = list()
    for fil in os.listdir(dir):
        if fil.lower().endswith('.jpeg'):
            try:
                img = cv.imread(dir + "/" + fil, 1)
            except cv.error:
                print("{} malformed!".format(fil))
                sys.exit()
            images.append(img)
    return images

def get_3D_hist(sub_img):
    """
    Take in a sub-image
    Get 3D histogram of the colors of the image and return it
    """
    M, N = sub_img.shape[:2]
    t = 4
    pixels = sub_img.reshape(M * N, 3)
    hist_3D, _ = np.histogramdd(pixels, (t, t, t))
    return hist_3D

def get_sub_imgs(img):
    """
    Take in an image
    Using b_w and b_h = 4, get all sub-images within the image
        (i.e. 25 blocks of equal size, evenly spaced from top left corner)
    Return the list of sub-images
    """
    H, W = img.shape[:2]
    b_w = b_h = 4
    del_w = W // (b_w + 1)
    del_h = H // (b_h + 1)

    sub_imgs = np.empty((5,5,del_h,del_w,3))
    for i in range(b_w+1):
        for j in range(b_h+1):
            w1 = i*del_w
            w2 = w1 + del_w
            h1 = j*del_h
            h2 = h1 + del_h
            sub_img = img[h1:h2, w1:w2]
            sub_imgs[i,j] = sub_img
    return sub_imgs

def get_desc_vec(sub_imgs):
    """
    Take in a list of sub-images
    For each sub-image:
        - Stack it with the sub-image next to it, the sub-image below it,
          and the sub-image below and to the right of it
          This will create a series of overlapping blocks since most sub-images
          will be used multiple times
    """
    desc_vec = np.empty((0))
    init = True
    for i in range(4):
        for j in range(4):
            block = np.vstack(
                (np.hstack((sub_imgs[i,j], sub_imgs[i+1, j])),
                np.hstack((sub_imgs[i,j+1], sub_imgs[i+1,j+1])))
            )
            if init == True:
                desc_vec = get_3D_hist(block)
                init = False
            else:
                desc_vec = np.hstack((desc_vec, get_3D_hist(block)))
    desc_vec = desc_vec.flatten()
    return desc_vec

if __name__ == "__main__":
    """
    Handle command line arguments
    """
    if len(sys.argv) != 2:
        print("Usage: {} train_dir".format(sys.argv[0]))
        sys.exit()
    else:
        train_dir_name = sys.argv[1]

    train_dir = list()
    try:
        train_dir = read_train_dir(train_dir_name)
    except FileNotFoundError:
        print("{} not found!".format(train_dir_name))
        sys.exit()

    """
    Create outfile for descriptors
    Go through each image found in the sub-directories
    Get the descriptor vector for each and append it to list
    Dump the descriptor vectors into outfile
    """
    outfile = open("{}/desc.txt".format(train_dir_name), "wb")
    desc_vec_list = []
    for i in range(len(train_dir)):
        for img in train_dir[i]:
            subimgs = get_sub_imgs(img)
            desc_vec = get_desc_vec(subimgs)
            desc_vec_list.append(desc_vec)
    pickle.dump(desc_vec_list, outfile)
    outfile.close()
