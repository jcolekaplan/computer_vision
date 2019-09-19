"""
    Jacob Kaplan
    kmeans.py
"""

import sys
import cv2 as cv
import numpy as np

def scale(img):
    """
    Take in image
    Reshape it to have width of 600 pixels
    Use OpenCV mean shift to recolor each pixel by shifting it towards
      the mode of a given radius of pixels
    Return recolored image
    """
    m, n = img.shape[:2]
    img = cv.resize(img, (600, int(600*(m/n))))
    shiftImg = cv.pyrMeanShiftFiltering(img, 50, 50, 2)
    return shiftImg

def cluster(img, K):
    """
    Take in image and integer K
    Use kmeans clustering to segment and image by color
    Return clustered image
    """
    input = np.float32(img.reshape((-1,3)))
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret, label, center = \
        cv.kmeans(input, K, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    cluster = center[label.flatten()]
    cluster = cluster.reshape((img.shape))
    return cluster

if __name__ == "__main__":
    """
    Handle command line arguments
    """
    if len(sys.argv) != 3:
        print("Usage: {} K".format(sys.argv[0]))
        sys.exit()
    else:
        inImgName = sys.argv[1]
        K = sys.argv[2]

    try:
        inImg = cv.imread(inImgName)
    except AttributeError:
        print("{} is not a valid image!".format(inImgName))
        sys.exit()

    try:
        K = int(K)
    except ValueError:
        print("K must be integer!")
        sys.exit()

    scaledImg = scale(inImg)
    clusterImg = cluster(scaledImg, K)
    name, ext = inImgName.split(".")
    cv.imwrite("{}_kmeans.{}".format(name, ext), clusterImg)
