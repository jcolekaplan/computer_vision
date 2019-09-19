"""
    Jacob Kaplan
    functions.py

    Implementations of functions called in align.py
"""

import os
import random
import cv2 as cv
import numpy as np

def findImages(directory):
    """
    Take in directory of images, open directory, read in each image,
    add to list of images, return list of images
    """
    images = list()
    for fil in os.listdir(directory):
        if fil.lower().endswith('.jpg'):
            try:
                img = cv.imread(directory + "/" + fil, 1)
            except cv.error:
                print("{} malformed!".format(fil))
                sys.exit()
            images.append((img, fil))
    return images

"""
1. Extract and match keypoints =================================================
"""
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

def outputMatchInfo(kps, name1, name2):
    """
    Take in keypoints and image names and output info about them
    """
    print("\nComparing {} and {}...".format(name1, name2))
    print("Applying ratio test...")
    print("{} matches found...".format(len(kps)))
    print("Drawing best matches...")

"""
2. Fundamental matrix ==========================================================
"""
def checkMatches(kps):
    """
    Take in keypoints
    If they are above 100, there are enough to continue, return True
    Else, False
    """
    if len(kps) >= 100:
        print("There are enough matches to continue...")
        return True
    else:
        print("There are not enough matches to continue.")
        return False

def getFundamental(kps1, kps2):
    """
    Take in two set of keypoints
    Convert to np arrays and get fundamental matrix using RANSAC algorithm
    Use mask of fundamental matrix to get keypoints that survived the estimation
    Return the inlying keypoints
    """
    kps1 = np.int32(kps1)
    kps2 = np.int32(kps2)
    funMat, mask = cv.findFundamentalMat(kps1, kps2, cv.RANSAC)
    ind = np.where(mask.ravel() == 1)[0]
    inl1 = kps1[ind]
    inl2 = kps2[ind]
    return inl1, inl2

def outputFundamentalInfo(kps, inl):
    """
    Take in keypoints before and after fundamental matrix estimation
    and output info about how many survived
    """
    print("Building fundamental matrix...")
    print("Of the {} best matches...".format(len(kps)))
    print("{} survived the fundamental matrix estimation...".format(len(inl)))
    print("Drawing inliers...")

"""
3. Check if there are enough inliers to continue ===============================
"""
def checkInliers(kps, inl):
    """
    Take in keypoints before and after fundamental matrix estimation
    If 80% or more survived, continue with program, return True
    Else, False
    """
    percentRemain = round(100* len(inl) / len(kps))
    print("{}% matches remain...".format(percentRemain))
    if percentRemain >= 80:
        print("That means the images are likely of the same scene...")
        return True
    else:
        print("The images are probably not of the same scene.")
        return False

"""
4. Estimate homography matrix ==================================================
"""
def getHomography(inl1, inl2):
    """
    Take in inliers of fundamental matrix estimation
    Use them to estimate homography matrix using RANSAC algorithm
    Use mask to get keypoints that survived homography estimation
    Return inliers and homography matrix
    """
    hmMat, mask = cv.findHomography(inl1, inl2, cv.RANSAC, 5.0)
    ind = np.where(mask.ravel() == 1)[0]
    hm1 = inl1[ind]
    hm2 = inl2[ind]
    return hm1, hm2, hmMat

def outputHomographyInfo(inl, hm):
    """
    Take in keypoints before and after homography matrix estimation
    Output info about how many remain
    """
    print("Building homography matrix ...")
    print("Of the {} original inliers...".format((len(inl))))
    print("{} survived the homography matrix estimation...".format(len(hm)))
    print("Drawing inliers...")

"""
5. Check alignment =============================================================
"""
def checkAlignment(inl, hm):
    """
    Take in keypoints before and after homography matrix estimation
    Get percent difference and if it's less than 30, return True
    Else, False
    """
    percentDiff = np.abs(round(100*(len(inl) - len(hm)) / len(inl)))
    print("Homography and fundamental matrix had a {}% difference in results..."\
            .format(percentDiff))
    if percentDiff <= 30:
        print("The images can be aligned...")
        return True
    else:
        print("The images cannot be aligned.")
        return False

"""
6. Build mosaic ================================================================
"""
def buildMosaic(img1, img2, homography):
    """
    Python Gods forgive me for this behemoth...

    Take in two images and their homography matrix
    Using the corners of the images, get the dimensions of the shape of the
      mosaic shape based on transform around the homography matrix
    Use the dimensions of the mosaic shape to get the coordinates of where to
      cut the original images (stitchX, stitchY)
    Use homography inverse to map image 2 to the new mosaic shape
    Get the overlap of the mapped image 2 and the original image 1
    Average the color of the overlapping section
    Paste image 1 onto the mapped image 2
    Return mosaic
    """
    
    print("Building mosaic...")
    # Original dimensions
    m1, n1 = img1.shape[:2]
    m2, n2 = img2.shape[:2]
    corners1 = np.asarray([[0,0], [0,m1], [n1,m1], [n1,0]]).reshape(-1,1,2)
    corners2 = np.asarray([[0,0], [0,m2], [n2,m2], [n2,0]]).reshape(-1,1,2)

    # Mosaic shape and dimensions
    transform = cv.perspectiveTransform(np.float32(corners2), homography)
    mosaicShape = np.vstack((corners1, transform))
    mCornersBot = np.min(mosaicShape, axis=0)
    mCornersTop = np.max(mosaicShape, axis=0)
    x1, y1 = mCornersTop[0,:2]
    x2, y2 = mCornersBot[0,:2]
    stitchX, stitchY = int(round(-1*x2)), int(round(-1*y2))

    # Map image 2 onto mosaic using invere homography
    affine = np.array([[1, 0, stitchX], [0, 1, stitchY], [0,0,1]])
    x1, x2 = int(round(x1)), int(round(x2))
    y1, y2 = int(round(y1)), int(round(y2))
    mosaic = cv.warpPerspective(img2, affine.dot(homography), (x1-x2, y1-y2))

    # Get overlapping section of mapped image 2 and original image 1
    mosaicZeros = np.zeros_like((mosaic))
    ms1 = mosaic.copy()
    ms2 = mosaicZeros.copy()
    ms1[ms1 > 0] = 1
    ms2[stitchY:stitchY+m1, stitchX:stitchX+n1] = 1
    overlap = ms1 + ms2
    mosaicTmp = mosaic.copy()
    mosaicZeros[stitchY:stitchY + m1, stitchX:stitchX + n1] = img1
    mosaic[stitchY:stitchY + m1, stitchX:stitchX + n1] = img1
    # Average the colors
    mosaic[np.where(overlap == 2)] = 0.5*mosaicZeros[np.where(overlap == 2)] \
                                    + 0.5*mosaicTmp[np.where(overlap == 2)]
    print("Mosaic built!")
    return mosaic

"""
   Drawing lines between keypoints =============================================
"""
def drawMatches(img1, img2, kps1, kps2):
    """
    Take in two images and their corresponding sets of keypoints
    Paste the two images onto a larger canvas
    Iterate through both sets of keypoints and extract coordinates
    Draw circle of random color around each keypoint onto new image
    Draw line of random color through each keypoint pair onto new image
    Return new image
    """
    m1, n1 = img1.shape[:2]
    m2, n2 = img2.shape[:2]
    draw = np.zeros((max(m1, m2), n1 + n2, 3), dtype="uint8")
    draw[0:m1, 0:n1] = img1
    draw[0:m2, n2:] = img2

    for i in range(len(kps1)):
        b = random.randint(0,255)
        g = random.randint(0,255)
        r = random.randint(0,255)

        pt1 = (int(kps1[i][0]), int(kps1[i][1]))
        pt2 = (int(kps2[i][0]) + n1, int(kps2[i][1]))

        cv.circle(draw, pt1, 1, (b,g,r), 1)
        cv.circle(draw, pt2, 1, (b,g,r), 1)
        cv.line(draw, pt1, pt2, (b, g, r), 2)

    return draw
