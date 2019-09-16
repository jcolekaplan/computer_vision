"""
    Jacob Kaplan
    best_focus.py

    Given a series of images (all in one folder) taken of the
    same scene, determine which image is focused the best.

                       ,@@@@@@@,
               ,,,.   ,@@@@@@/@@,  .oo8888o.
            ,&%%&%&&%,@@@@@/@@@@@@,8888\88/8o
           ,%&\%&&%&&%,@@@\@@@/@@@88\88888/88'
           %&&%&%&/%&&%@@\@@/ /@@@88888\88888'
           %&&%/ %&%%&&@@\ V /@@' `88\8 `/88'
           `&%\ ` /%&'    |.|        \ '|8'
               |o|        | |         | |
               |.|        | |         | |
            \\/ ._\//_/__/  ,\_//__\\/.  \_//__/_
"""

import sys
import cv2 as cv
import numpy as np
from os import listdir

def findImages(directory):
    """
    Take in directory of images, open directory, read in each image,
    add to list of images and names, return list of images and image names
    """
    images = list()
    names = list()
    for fil in listdir(directory):
        if fil.lower().endswith('.jpg'):
            try:
                img = cv.imread(directory + "/" + fil, 0)
            except cv.error:
                print("{} malformed!".format(fil))
                sys.exit()
            images.append(img)
            names.append(fil)
    return images, names

def getGradientMag(image):
    """
    Take in an image
    Calculate Sobel derivatives in the x and y directions
    Get magnitude of gradient of entire image
    Return magnitude
    """
    sobelX = cv.Sobel(image,cv.CV_64F,1,0)
    sobelY = cv.Sobel(image,cv.CV_64F,0,1)
    M,N = image.shape
    gradientMag = np.sum(sobelX**2 + sobelY**2) / (M*N)
    return gradientMag

def outGradients(imgGrads):
    """
    Take in list of stats of each image (name, gradient)
    Output stats of each image
    Output best focused image (largest gradient)
    """
    largestGrad = imgGrads[0]
    for inf in imgGrads:
        if inf[1] > largestGrad[1]:
            largestGrad = inf
        print("{}: {:.2f}".format(inf[0], inf[1]))
    print("Image {} is best focused.".format(largestGrad[0]))

if __name__ == "__main__":
    """
    Handle command line arguments
    """
    if len(sys.argv) != 2:
        print("Usage: {} imageDirectory".format(sys.argv[0]))
        sys.exit()
    else:
        directory = sys.argv[1]

    try:
        images, names = findImages(directory)
    except FileNotFoundError:
        print("Directory does not exist!")
        sys.exit()

    """
    Get magnitudes of each image, put in list, output stats
    """
    imgGradients = list()
    for i in range(len(images)):
        grad = getGradientMag(images[i])
        imgGradients.append((names[i],grad))
    outGradients(sorted(imgGradients))
