"""
    Jacob Kaplan
    functions.py
    
    Helper functions for all the functionality described in motion.py

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
            I'm here to help with functionality
"""

import random
import cv2 as cv
import numpy as np

"""
================================================================================
============== 1. Calculate optical flow and draw motion vectors  ==============
================================================================================
"""
def calcOptFlow(imOne, imTwo):
    """
    Takes in two images
    Converts both to grayscale
    Calculate Shi-Tomasi corner points in the first image
    Track those points in the second image and use Lucas-Kanade optical flow
        to get motion vectors of those points as they move from first image
        to second
    Filter out points that are too close to one another
    Return the points
    """

    imOneGr = cv.cvtColor(imOne, cv.COLOR_BGR2GRAY)
    imTwoGr = cv.cvtColor(imTwo, cv.COLOR_BGR2GRAY)

    imOnePts = cv.goodFeaturesToTrack(imOneGr, 250, 0.01, 10, blockSize = 10)
    imTwoPts, st, _ = cv.calcOpticalFlowPyrLK(imOneGr, imTwoGr, 
                                              imOnePts, None, winSize = (15, 15),
                                              maxLevel = 2)
    imOnePts = imOnePts[st==1]
    imTwoPts = imTwoPts[st==1]
    imOnePts, imTwoPts = filterDist(imOnePts, imTwoPts)
    return imOnePts, imTwoPts

def filterDist(imOnePts, imTwoPts):
    """
    Take in two sets of points
    Goes through each corresponding pair of points and checks to see that each
        x and y value are more than 1 pixel away from the other x and y
        value, respectively
    Returns the filtered sets of points
    """
    imOnePtsFt = []
    imTwoPtsFt = []
    for i in range(len(imOnePts)):
        x1, y1 = imOnePts[i]
        x2, y2 = imTwoPts[i]
        if np.abs(x1 - x2) > 1 and np.abs(y1 - y2) > 1:
            imOnePtsFt.append([x1, y1])
            imTwoPtsFt.append([x2, y2])
    imOnePtsFt = np.int32(imOnePtsFt)
    imTwoPtsFt = np.int32(imTwoPtsFt)
    return imOnePtsFt, imTwoPtsFt

def drawMotionVectors(imTwo, imOnePts, imTwoPts):
    """
    Take in one image and two sets of points
    Draw all the points in white on the image and a line connecting each
        corresponding set of points
    Return the drawn on image
    """
    draw = imTwo.copy()
    numPts = imOnePts.shape[0]
    for i in range(numPts):
        x1, y1 = imOnePts[i]
        x2, y2 = imTwoPts[i]
        cv.circle(draw, (x1, y1), 2, random_bgr(), 1)
        cv.circle(draw, (x2, y2), 3, random_bgr(), 1)
        cv.line(draw, (x1, y1), (x2, y2), random_bgr(), 2)
    return draw

def random_bgr():
    b,g,r = random.randint(0,255),random.randint(0,255),random.randint(0,255)
    return (b,g,r)
"""
================================================================================
======================= 2. Determine if camera is moving  ======================
================================================================================
"""

def isCameraMoving(imOnePts):
    """
    Takes in a set of points
    Returns true if there are more than 20 points. Otherwise, false

    This works because the points in the first image are filtered after getting
        the optical flow in the second image.
    With a non-moving camera, the vast majority of the points stay still thus
        this number would be low. Otherwise, it will be very high because most
        of the points wind up moving
    """
    if imOnePts.shape[0] >= 100:
        print("Camera is moving!")
        return True
    else:
        print("Camera is not moving!")
        return False

"""
================================================================================
===================== 3. Find and draw focus of expansion ======================
================================================================================
"""
def getFocusOfExpansion(imOne, imTwo):
    """
    Takes two images
    Matches SIFT keypoints between the two images
    Gets the homography matrix inliers after homography matrix estimation of
        these keypoints
    Estimate the center (mean) of these inliers using linear regression
    Return the center (focus of expansion)
    """
    kps1, kps2 = matchKeyPoints(imOne, imTwo)
    kps1, kps2 = np.int32(kps1), np.int32(kps2)
    hm1, hm2 = getHomography(kps1, kps2)
    xVals, yVals = hm1[:,0], hm1[:,1]
    fit = np.polyfit(xVals, yVals, 2)
    func = np.poly1d(fit)
    maxY, maxX = 0, 0
    for i in range(1000):
        if func(i) >= maxY:
            maxY, maxX = int(func(i)), i
    return (maxX, maxY)

def matchKeyPoints(img1, img2):
    """
    Take in two images
    Get a set of keypoints and descriptors for each image
    Matches both sets of keypoints with brute-force matcher
    Perform ratio test to get best matches
    Extract the keypoints that survived ratio test and return them
    """
    kps1, dsc1 = getSIFT(img1)
    kps2, dsc2 = getSIFT(img2)
    allMatches = matchAll(dsc1, dsc2)
    ratioMatches = ratioTest(kps1, kps2, allMatches)
    kps1, kps2 = extractKps(kps1, kps2, ratioMatches)
    return kps1, kps2

def getSIFT(img):
    """
    Take in image
    Return SIFT keyPoints and descriptors
    """
    sift = cv.xfeatures2d.SIFT_create()
    keyPoints, descriptors = sift.detectAndCompute(img, None)
    return keyPoints, descriptors

def matchAll(dsc1, dsc2):
    """
    Take in two sets of descriptors for keypoints
    Use brute-force matcher to get matches
    Return matches
    """
    bfMatcher = cv.BFMatcher()
    matches = bfMatcher.knnMatch(dsc1,dsc2,k=2)
    return matches

def ratioTest(kps1, kps2, matches):
    """
    Take in two sets of keypoints and the set of matches between them
    Perform ratio test to get best matches (60% threshold)
    Return surviving matches
    """
    ratioMatches = list()
    for match in matches:
        m1, m2 = match
        if m1.distance < 0.60*m2.distance:
            ratioMatches.append(m1)
    return ratioMatches

def extractKps(kps1, kps2, ratioMatches):
    """
    Take in two sets of keypoints and the matches that survived ratio test
    Extract all the keypoints in the set of ratio matches
    Return extracted keypoints
    """
    newKps1 = list()
    newKps2 = list()
    for match in ratioMatches:
        newKps1.append(kps1[match.queryIdx].pt)
        newKps2.append(kps2[match.trainIdx].pt)
    return newKps1, newKps2

def getHomography(kps1, kps2):
    """
    Take in best-matching keypoints
    Use them to estimate homography matrix using RANSAC algorithm
    Use mask to get keypoints that survived homography estimation
    Return inliers
    """
    hmMat, mask = cv.findHomography(kps1, kps2, cv.RANSAC, 5.0)
    ind = np.where(mask.ravel() == 1)[0]
    hm1 = kps1[ind]
    hm2 = kps2[ind]
    return hm1, hm2

def drawFocusOfExpansion(imTwo, FoE, pts):
    """
    Take in image and focus of expansion
    Draws focus of expansion as circle of random color
    Return drawn on image
    """
    draw = imTwo.copy()
    cv.circle(draw, FoE, 5, random_bgr(), 3)
    for pt in pts:
        x, y = pt
        cv.line(draw, (x, y), FoE, random_bgr(), 2)
    return draw 

"""
================================================================================
============== 4. Find and draw rectangle around moving objects ================
================================================================================
"""

def motionDetector(imOne, imTwo):
    """
    Takes in two images
    Converts both to grayscale
    Gets the absolute difference between the two
    If the difference between the images is great enough, find the contours
        representing moving objects between the two images
    Return the contours
    """
    imOneGr = cv.cvtColor(imOne, cv.COLOR_BGR2GRAY)
    imTwoGr = cv.cvtColor(imTwo, cv.COLOR_BGR2GRAY)

    imTwoGr = cv.GaussianBlur(imTwoGr, (15, 15), 0)

    diff = cv.absdiff(imOneGr, imTwoGr)
    thresh = cv.threshold(diff, 100, 255, cv.THRESH_BINARY)[1]
    thresh = cv.dilate(thresh, None, iterations=2)

    contours = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[1]
    return contours

def drawMovingObjects(imTwo, contours):
    """
    Take in image and contours from the motion detector
    If the contour is not too large, draw a bounding box around it of a 
        random color
    Return the drawn on image
    """
    for contour in contours:
        if cv.contourArea(contour) >= 1000:
            x1, y1, x2, y2 = cv.boundingRect(contour)
            b,g,r = random.randint(0,255), random.randint(0,255), random.randint(0,255)
            cv.rectangle(imTwo, (x1, y1), (x1+x2, y1+y2), (b,g,r), 2)
    return imTwo
