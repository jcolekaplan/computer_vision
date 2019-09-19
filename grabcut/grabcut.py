"""
    Jacob Kaplan
    grabcut.py

    Explore the GrabCut function in OpenCV
"""

import sys
import random
import cv2 as cv
import numpy as np

def makeTuple(rect):
    """
    Take in rectangle coordinates as list and return as tuple
    """
    rectTuple = (rect[0], rect[1], rect[2], rect[3])
    return rectTuple

def outerRect(img, rect):
    """
    Take in image and its bounding rectangle
    Return the grabcut of the image bounded by rect
    """
    rect = makeTuple(rect)
    mask = np.zeros(img.shape[:2], np.uint8)
    cutImg = grabCut(img, rect, mask)
    return cutImg

def innerRect(img, rects, maskedImg, mask):
    """
    Take in image, a list of rectangles, the grabcut of the outer rectangle,
      and the mask the initial grabcut produced
    Go through each rectangle, update the mask
    Return final grabcut
    """
    maskedImg = cv.cvtColor(maskedImg, cv.COLOR_BGR2GRAY)
    for rect in rects:
        x1, y1, x2, y2 = makeTuple(rect)
        maskedImg[y1:y2, x1:x2] = 0
        mask[maskedImg == 0] = 0
        mask[maskedImg == 255] = 1
    cutImg = grabCut(img, None, mask)
    return cutImg

def grabCut(img, rect, mask):
    """
    Take in image, a bounding rectangle, and a mask
    Take grabcut of image within rectangle
    Return the result of the grabCut and the mask it produces
    """
    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)

    if rect is None:
        mask, bgdModel, fgdModel = \
            cv.grabCut(inImg, mask, None, bgdModel, fgdModel, 5, cv.GC_INIT_WITH_MASK)
    else:
        cv.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv.GC_INIT_WITH_RECT)

    mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
    maskedImg = img*mask2[:,:,np.newaxis]
    return maskedImg, mask

def drawRectangles(img, rects):
    """
    Take in image and list of rectangle coordinates
    Return image with rectangles of random colors drawn on it
    """
    for rect in rects:
        x1, y1, x2, y2 = makeTuple(rect)
        b = random.randint(0,255)
        g = random.randint(0,255)
        r = random.randint(0,255)

        pt1 = (x1, y1)
        pt2 = (x2, y2)
        cv.rectangle(img, pt1, pt2, (b,g,r), 2)
    return img

if __name__ == "__main__":
    """
    Handle command line arguments
    """
    if len(sys.argv) != 3:
        print("Usage: {} points".format(sys.argv[0]))
        sys.exit()
    else:
        inImgName = sys.argv[1]
        pointsFileName = sys.argv[2]

    try:
        inImg = cv.imread(inImgName)
    except AttributeError:
        print("{} is not a valid image!".format(inImgName))
        sys.exit()

    try:
        points = np.loadtxt(pointsFileName, dtype=np.uint16)
    except ValueError:
        print("Malformed points file: {}, must be numbers".format(pointsFileName))
        sys.exit()
    except OSError:
        print("Points file not found!")
        sys.exit()

    if len(points) < 2:
        print("Need points for inner and outer rectangles!")
        sys.exit()

    outerRectangle = points[0]
    innerRectangles = points[1:]

    outerCut, mask = outerRect(inImg, outerRectangle)
    innerCut, mask = innerRect(inImg, innerRectangles, outerCut, mask)
    drawRect = drawRectangles(inImg, points)

    name, ext = inImgName.split(".")
    cv.imwrite("{}_outer.{}".format(name, ext), outerCut)
    cv.imwrite("{}_inner.{}".format(name, ext), innerCut)
    cv.imwrite("{}_rect.{}".format(name, ext), drawRect)
