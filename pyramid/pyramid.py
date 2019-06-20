"""
    Jacob Kaplan
    pyramid.py

    Purpose: Write a Python program that takes a single image as input and creates a
    new image that stores the original, a half-sized copy, a quarter-sized copy,
    etc., side-by-side.
                                           _L/L
                                         _LT/l_L_
                                       _LLl/L_T_lL_
                   _T/L              _LT|L/_|__L_|_L_
                 _Ll/l_L_          _TL|_T/_L_|__T__|_l_
               _TLl/T_l|_L_      _LL|_Tl/_|__l___L__L_|L_
             _LT_L/L_|_L_l_L_  _'|_|_|T/_L_l__T _ l__|__|L_
           _Tl_L|/_|__|_|__T _LlT_|_Ll/_l_ _|__[ ]__|__|_l_L_
       _ _LT_l_l/|__|__l_T _T_L|_|_|l/___|__ | _l__|_ |__|_T_L_  __

                           nn_r   nn_r                 __
                     __   /l(\   /l)\      nn_r
               __                         /\(\    __

      3,000 years later and we're still building pyramids...
"""
import sys
import cv2 as cv
import math as mt
import numpy as np

def createPyramid(img):
    """
    Take in image as input
    Creates a black background based on its height and the calculated upper bound
    based on width
    Creates a list of images, each one half the size of the last
    Places those images on the background
    Returns the background with the placed images
    """
    M = img.shape[0]
    N = img.shape[1]
    background = np.full((M, calculateUpperBound(N), 3), [0,0,0])
    halvedImages = halveImages(img)
    pyramid = placeImages(background, halvedImages)
    return pyramid

def calculateUpperBound(N):
    """
    Takes in width as input, int divide by 2 until size minimum is reached and
    sums them to get the upper bound for width
    Return upper bound
    """
    halveBound = N
    upperBound = 0
    while (halveBound >= 20):
        upperBound += halveBound
        halveBound //= 2
    return upperBound

def halveImages(img):
    """
    Takes image as input
    While both height and width are greater than 20, halve the image in both
    dimensions and add it to list of halved images
    Return list of halved images
    """
    M = img.shape[0]
    N = img.shape[1]
    halvedImg = img
    halvedImages = list()
    halvedImages.append(halvedImg)
    while (M//2 >= 20 and N//2 >= 20):
        M //= 2
        N //= 2
        halvedImg = cv.resize(halvedImg, (N, M))
        halvedImages.append(halvedImg)

    return halvedImages

def placeImages(background, halvedImages):
    """"
    Takes in black background and list of halved images as input
    Places each image in its corresponding location on the background and output
    stats about the upper left bound where its placed and the image's size
    Outputs final stats and returns final image
    """
    M = 0
    N = 0
    for img in halvedImages:
        M2 = img.shape[0]
        N2 = img.shape[1]
        background[M//2:(M//2)+M2, N:N+N2] = img
        print("Copy starts at ({}, {}) image shape ({}, {}, {})"\
        .format(M//2, N, M2, N2, img.shape[2]))
        M += mt.ceil(M2/2)
        N += N2

    print("Final shape {}".format(background.shape))
    return background

if __name__ == "__main__":
    """
    Handle command line arguments
    """
    if len(sys.argv) != 3:
        print("Correct usage: p1_pyramid inImg outImg")
        sys.exit()
    else:
        inImgName = sys.argv[1]
        outImgName = sys.argv[2]

    try:
        inImg = cv.imread(inImgName).astype(np.float64)
    except AttributeError:
        print("{} not in a valid format! Must be .jpg!".format(inImgName))
        sys.exit()

    outImg = createPyramid(inImg)
    try:
        cv.imwrite(outImgName, outImg)
    except cv.error:
        print("Output file: {} not in a valid format! Must be .jpg!".format(outImgName))
        sys.exit()
