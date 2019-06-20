"""
Author:   Jacob Kaplan
File:     p2_shade.py

Purpose: Take an image and shade it proportionally to its size and from a specified direction (left, right, top, bottom, or center)
         The direction specified is the simulated source of light causing the shading

                             ,
                            /|      __
                           / |   ,-~ /
                          Y :|  //  /
                          | jj /( .^
                          >-"~"-v"
                         /       Y
                        jo  o    |
                       ( ~T~     j
                        >._-' _./
                       /   "~"  |
                      Y     _,  |
                     /| ;-"~ _  l
                    / l/ ,-"~    \
                    \//\/      .- \
                     Y        /    Y    
                     l       I     !
                     ]\      _\    /"\
                    (" ~----( ~   Y.  )
                ~~~~~~~~~~~~~~~~~~~~~~~~~~
                There are no carrots. I checked.
"""

import sys
import cv2 as cv
import numpy as np
import math as mt
import os
os.getcwd()

def generateMultiplier(img, direction):
    """
    Take in image and direction, find dimensions and call appropriate function to generate multiplier
    Multiplier generators return an array of distances from the specified direction.
    i.e. the left multiplier returns an array with a vector of zeros in the leftmost column
    Then, normalize the multipliers so they are at the same scale but from 0 to 1
    Return normalized multiplier
    """
    M = img.shape[0]
    N = img.shape[1]

    multiplier = None
    if direction == "right":
        multiplier = rightDist(M, N)
    elif direction == "left":
        multiplier =  leftDist(M, N)
    elif direction == "top":
        multiplier =  topDist(M, N)
    elif direction == "bottom":
        multiplier =  bottomDist(M, N)
    elif direction == "center":
        multiplier =  centerDist(M, N)
    else:
        sys.exit()
    # Normalize
    multiplier = multiplier / multiplier.max()
    multiplier = 1 - multiplier
    return multiplier

def leftDist(M, N):
    """
    Take dimensions, create array from 0 to N, tile array
    e.g. M = 3 N = 4
    0 1 2 3
    0 1 2 3
    0 1 2 3
    Return array of distances from left
    """
    dist = np.arange(N)
    distArray = np.tile(dist, (M,1))
    return distArray

def rightDist(M, N):
    """
    Take dimensions, call leftDist and rotate twice
    e.g. M = 3 N = 4
    3 2 1 0
    3 2 1 0
    3 2 1 0
    Return array of distances from right
    """
    return np.rot90(leftDist(M, N), 2)

def topDist(M, N):
    """
    Take dimensions, create array from 0 to M, tile array, and transpose
    e.g. M = 3 N = 4
    0 0 0 0
    1 1 1 1
    2 2 2 2
    Return array of distances from top
    """
    dist = np.arange(M)
    distArray = np.tile(dist, (N, 1))
    distArray = distArray.transpose()
    return distArray

def bottomDist(M, N):
    """
    Take dimensions, call topDist and rotate twice
    e.g. M = 3 N =4
    2 2 2 2
    1 1 1 1
    0 0 0 0
    Return array of distances from bottom
    """
    return np.rot90(topDist(M, N), 2)

def centerDist(M, N):
    """
    Take dimensions, call rightDist and topDist for distances from the x and y directions
    Use Euclid distance formula to create matrix where each value is proportional to 
    its distance from the center    
    Return array of distances from center
    """
    xDist = rightDist(M, N)
    yDist = topDist(M, N)
    distArray = np.sqrt((xDist - xDist[M//2, N//2])**2 + (yDist - yDist[M//2, N//2])**2)
    return distArray    

def shade(img, multiplier):
    """
    Take image and the normalized multiplier. Use Numpy magic to multiply each pixel in the image with its corresponding number in the multiplier
    Return shaded image
    """
    newImg = np.einsum('ijk, ij->ijk', img, multiplier)
    return newImg

def outputStats(mult):
    """
    Print stats about multiplier
    """
    M = mult.shape[0]
    N = mult.shape[1]
    rowPoints = [0, M//2, M-1]
    colPoints = [0, N//2, N-1]
    for rpt in rowPoints:
        for cpt in colPoints:
            print("({},{}) {:.3f}".format(rpt, cpt, mult[rpt, cpt]))

if __name__ == "__main__":
    """
    Handle command line arguments
    """

    if len(sys.argv) != 4:
        print("Correct usage: {} inputImage outputFile direction".format(sys.argv[0]))
        sys.exit()
    else:
        imgIn = sys.argv[1]
        imgOut = sys.argv[2]
        direction = sys.argv[3]
    
    # Weird Submitty thing
    if imgOut.find(".jpg") == -1:
        imgOut += ".jpg"

    # Read image, generate multiplier, shade image
    try:
        img = cv.imread(imgIn).astype(np.float64)
    except AttributeError:
        print("Image not found or not valid format!")
        sys.exit()

    multiplier = generateMultiplier(img, direction)
    shadedImg = shade(img, multiplier)
    # Concatenate shaded image with the original, output stats, write out the combined image
    out = np.concatenate((img, shadedImg), axis=1)
    outputStats(multiplier)
    try:
        cv.imwrite(imgOut, out)
    except cv.error:
        print("Output file not valid!")
        sys.exit()
