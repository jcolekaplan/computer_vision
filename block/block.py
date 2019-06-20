"""
Author:   Jacob Kaplan
Date: September 14, 2018
File:     p1_block.py

Purpose: Reading in an image, calculating average intensity per block based on number of rows and columns given,
         building an image from those blocks based on block size given, and converting that into a binary image based
         on a calculated threshold.



                       ---_ ......._-_--.
                      (|\ /      / /| \  \
                      /  /     .'  -=-'   `.
                     /  /    .'             )
                   _/  /   .'        _.)   /
                  / o   o        _.-' /  .'
                  \          _.-'    / .'*|
                   \______.-'//    .'.' \*|
                    \|  \ | //   .'.' _ |*|
                     `   \|//  .'.'_ _ _|*|
                      .  .// .'.' | _ _ \*|
                      \`-|\_/ /    \ _ _ \*\
                       `/'\__/      \ _ _ \*\
                      /^|            \ _ _ \*
                     '  `             \ _ _ \      Hssssss...it's nice to be using Python again...
                                       \_

"""
import sys
import math as mt
import cv2 as cv
import numpy as np

def downsizeAvg(image, m, n):
    """
    Takes in original greyscale image and desired dimensions. Calculates scale factor then iterates
    through each block of the downsized image, converting the original image to its average intenstiy per block
    Return downsized image
    """
    imgM = image.shape[0]
    imgN = image.shape[1]
    sM = imgM/m
    sN = imgN/n
    
    for i in range(0,m):
        # Boundaries of the block as specified in the homework instructions
        xBound1 = mt.floor(i * sM)
        xBound2 = mt.floor((i+1) * (sM))
    
        for j in range(0,n):
            # Boundaries of the block as specified in the homework instructions
            yBound1 = mt.floor(j * sN)
            yBound2 = mt.floor((j+1)*(sN))
            
            # Find average per block and change all pixels in boundaries to average
            imageBound = image[xBound1:xBound2, yBound1:yBound2]
            avg = np.average(imageBound)
            image[xBound1:xBound2, yBound1:yBound2] = avg
    
    return cv.resize(image, (n*b, m*b))

def downsizeBin(image):
    """
    Takes in the downsized image (greyscale pixelated), finds its size (since it's changed)
    Changes pixels over threshold (median) to white and under to black 
    Returns binarized image
    """
    M = image.shape[0]
    N = image.shape[1]
    med = np.median(image)
    image[image >= med] = 255
    image[image < med] = 0    
    return image

def outputStats(m, n, b, image):
    """
    Output stats on the downsized images
    """
    print("Downsized images are ({}, {})".format(m//b, n//b))
    print("Block images are ({}, {})".format(m, n))
    print("Average intensity at ({}, {}) is {:.2f}".format(m//4//b, n//4//b, image[m//4, n//4]))
    print("Average intensity at ({}, {}) is {:.2f}".format(m//4//b, 3*n//4//b, image[m//4, (3*n)//4]))
    print("Average intensity at ({}, {}) is {:.2f}".format(3*m//4//b, n//4//b, image[3*m//4, n//4]))
    print("Average intensity at ({}, {}) is {:.2f}".format(3*m//4//b, 3*n//4//b, image[3*m//4, 3*n//4]))
    print("Binary threshold: {:.2f}".format(np.median(image)))

def rename(imgName, char):
    """
    Helper function for renaming output files
    """
    title = imgName.split(".jpg")[0]
    newTitle = "{}_{}.jpg".format(title, char)
    return newTitle

if __name__ == "__main__":
    """
    Handle command line arguments
    """
    if len(sys.argv) != 5:
        print("Correct usage: {} inputImage rows cols blockSize".format(sys.argv[0]))
        sys.exit()
    else:
        imgName = sys.argv[1]
        m = sys.argv[2]
        n = sys.argv[3]
        b = sys.argv[4]
    
    try:
        m = int(m)
        n = int(n)
        b = int(b)
    except ValueError:
        print("Rows, cols, and block size must be ints!")
        sys.exit()

    # Read image as greyscale of type float64
    try:
        img = cv.imread(imgName, 0).astype(np.float64)
    except AttributeError:
        print("Image not found or not valid format!")
        sys.exit()

    # Create first downsized image (pixelated greyscale), output stats, and write out new image
    avgImg = downsizeAvg(img, m, n)
    outputStats(m*b, n*b, b, avgImg)
    cv.imwrite(rename(imgName, "g"), avgImg)
    print("Wrote image {}".format(rename(imgName, "g")))
    
    # Create second downsized image (binary) and write out new image
    binImg = downsizeBin(avgImg)
    cv.imwrite(rename(imgName, "b"), binImg)
    print("Wrote image {}".format(rename(imgName, "b")))
