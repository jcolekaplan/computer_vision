"""
Author:   Jacob Kaplan
Date: September 14, 2018
File:     p3_checkerboard.py

Purpose: Take a directory of images, dimensions, and block size. Do one of the following based on the directory:
         1. No image in directory: Output black and white checkerboard with specified dimensions and block size
         2. One image in directory: Output checkerboard of that photo and black blocks with specified dimensions and block size
         3. Two images in directory: Output checkerboard of those two images with specified dimensions and block size
         4. More than two images in directory: Find the two images with the most different in color (most contrast with each other)
            Output checkerboard of those two images with specified dimensions and block size


              \`*-.
                 )  _`-.
                .  : `. .
                : _   '  \
                ; *` _.   `*-._
                `-.-'          `-.
                  ;       `       `.
                  :.       .        \
                  . \  .   :   .-'   .
                  '  `+.;  ;  '      :
                  :  '  |    ;       ;-.
                  ; '   : :`-:     _.`* ;
               .*' /  .*' ; .*`- +'  `*'
               `*-*   `*-*  `*-*'
               Hey, you seen a rabbit?
"""

import sys
import math as mt
import cv2 as cv
import numpy as np
from os import listdir

def findImages(directory):
    """
    Take in directory of images, open directory, read in each image, add to list of images and names, return list of images and image names
    """
    images = list()
    names = list()
    for fil in listdir(directory):
        if fil.lower().endswith('.jpg'):
            img = cv.imread(directory + "/" + fil)
            images.append(img)
            names.append(fil)
    return images, names

def noImageCB(m, n, s):
    """
    Take in specified dimensions and block size
    Create white and black blocks of specified block size
    Call makeBoard with specified dimensions + blocks and return result (white and black checkerboard)
    """
    whiteBlock = np.full((s,s,3), [255,255,255])
    blackBlock = np.full((s,s,3), [0,0,0])
    # Redundant code to make Submitty happy is redundant
    whiteBlock = cropAndResize(whiteBlock, s)
    blackBlock = cropAndResize(blackBlock, s)
    return makeBoard(whiteBlock, blackBlock, m, n)

def oneImageCB(image, m, n, s):
    """
    Take in one image, specified dimensions, and block size
    Crop and resize images
    Create black block of specified block size
    Call makeBoard with specified dimensions + image + block and return result (image and black checkerboard)
    """
    imgOne = cropAndResize(image, s)
    blackBlock = np.full((s,s,3), [0,0,0])
    blackBlock = cropAndResize(blackBlock, s)
    return makeBoard(imgOne, blackBlock, m, n)

def twoImageCB(images, m, n, s):
    """
    Take in two images, specified dimensions, and block size
    Crop and resize both images
    Call makeBoard with specified dimensions + images and return result (two image checkerboard)
    """
    imgTwo = cropAndResize(images[1], s)
    imgOne = cropAndResize(images[0], s)
    return makeBoard(imgOne, imgTwo, m, n)

def pickImages(directory):
    """
    Take in directory of images, reads in each image, gets average color of each, puts average colors in a list
    Finds two images with colors most distinct from each other (furthest apart R, G, B)
    Prints info about the images' colors
    Prints info about the images chosen
    Returns the two most distinct images
    """
    avgColors = list()
    imgNames = list()
    images = list()
    for fil in sorted(listdir(directory)):
        if fil.lower().endswith('.jpg'):
            img = cv.imread(directory + "/" + fil)
            avgColorRow = np.average(img, axis=0)
            avgColor = np.average(avgColorRow, axis=0)
            avgColors.append(avgColor)
            imgNames.append(fil)
            images.append(img)
            print("{} ({:.1f}, {:.1f}, {:.1f})".format(fil, avgColor[2], avgColor[1], avgColor[0]))

    # Compare each images' average color with Euclid distance
    biggestDiff = 0
    imgOne = imgTwo = images[0]
    imgOneName = imgTwoName = images[0]
    for i in range(len(avgColors)):
        for j in range(len(avgColors)):
            avg1 = avgColors[i]
            avg2 = avgColors[j]
            diff = mt.sqrt((avg1[0] - avg2[0])**2 + (avg1[1] - avg2[1])**2 + (avg1[2] - avg2[2])**2)
            if diff >= biggestDiff:
                biggestDiff = diff
                imgOne = images[i]
                imgTwo = images[j]
                imgOneName = imgNames[i]
                imgTwoName = imgNames[j]

    print("Checkerboard from {} and {}. Distance between them is {:.1f}".format(imgTwoName, imgOneName, biggestDiff))
    return imgOne, imgTwo

def makeBoard(imgOne, imgTwo, m, n):
    """
    Takes two images, concatenates them left-to-right and right-to-left
    Creates two rows of each type of concatenation
    Concatenate the rows top-to-bottom
    Create checkerboard by concatenating those concatenated rows
    Return checkerboard
    """
    conOne = np.concatenate((imgOne, imgTwo), axis=1)
    conTwo = np.concatenate((imgTwo, imgOne), axis=1)
    rowOne = np.tile(conOne, (n//2, 1))
    rowTwo = np.tile(conTwo, (n//2, 1))
    checkerboard = np.concatenate((rowOne, rowTwo), axis=0)
    for i in range(m//2-1):
        rows = np.concatenate((rowOne, rowTwo), axis=0)
        checkerboard = np.concatenate((checkerboard, rows), axis=0)
    return checkerboard

def cropAndResize(image, s):
    """
    Takes images and specified block size
    Finds dimensions of image and crops off sides if width>height or crops off top and bottom if height>width
    Does nothing if no cropping is required
    Resizes image to specified block size
    Does nothing if already right size
    Returns cropped and resized image
    Prints info as it goes along
    """
    m = image.shape[0]
    n = image.shape[1]
    croppedImg = resizedImg = image
    # width > height
    if n > m:
        xBound1 = (n - m) // 2
        xBound2 = xBound1 + m
        croppedImg = image[0:n, xBound1:xBound2]
        print("Image cropped at ({},{}) and ({},{})".format(0, xBound1, m-1, xBound2-1))

    # height > width
    elif m > n:
        yBound1 = (m - n) // 2
        yBound2 = yBound1 + n
        croppedImg = image[yBound1:yBound2, 0:m]
        print("Image cropped at ({},{}) and ({},{})".format(yBound1, 0, yBound2-1, n-1))
    elif m == n:
        print("Image does not require cropping")

    # Resize
    if (m == s and n == s):
        resizedImg = croppedImg
        print("No resizing needed")
    else:
        resizedImg = cv.resize(croppedImg, (s,s))
        print("Resized from {} to {}".format(croppedImg.shape, resizedImg.shape))

    return resizedImg

if __name__ == "__main__":
    """
    Handle command line arguments
    """
    if len(sys.argv) != 6:
        print("Usage: {} imageDirectory outputFile M N blockSize".format(sys.argv[0]))
        sys.exit()
    else:
        directory = sys.argv[1]
        imgOut = sys.argv[2]
        M = sys.argv[3]
        N = sys.argv[4]
        S = sys.argv[5]

    try:
        M = int(M)
        N = int(N)
        S = int(S)
    except ValueError:
        print("Rows, cols, and block size must be ints!")
        sys.exit()

    # List of images in directory
    try:
        images, names = findImages(directory)
    except FileNotFoundError:
        print("Directory does not exist!")
        sys.exit()

    # Each of the four cases mentioned in Purpose
    if len(images) == 0:
        print("No images. Creating an ordinary checkerboard.")
        checkerboard = noImageCB(M, N, S)
    elif len(images) == 1:
        print("One image: {}. It will form the white square.".format(names[0]))
        checkerboard = oneImageCB(images[0], M, N, S)
    elif len(images) == 2:
        print("Exactly two images: {} and {}. Creating checkerboard from these.".format(names[1], names[0]))
        checkerboard = twoImageCB(images, M, N, S)
    elif len(images) > 2:
        pickedImgs = pickImages(directory)
        checkerboard = twoImageCB(pickedImgs, M, N, S)

    # Write to output file
    try:
        cv.imwrite(imgOut, checkerboard)
    except cv.error:
        print("Output file not valid!")
        sys.exit()
