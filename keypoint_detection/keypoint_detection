"""
Jacob Kaplan
keypoint_detection.py

Check how consistent the results of Harris keypoint detection and SIFT keypoint
detection with each other are.
                                  _/`.-'`.
                        _      _/` .  _.'
               ..:::::.(_)   /` _.'_./
             .oooooooooo\ \o/.-'__.'o.
            .ooooooooo`._\_|_.'`oooooob.
          .ooooooooooooooooooooo&&oooooob.
         .oooooooooooooooooooo&@@@@@@oooob.
        .ooooooooooooooooooooooo&&@@@@@ooob.
        doooooooooooooooooooooooooo&@@@@ooob
        doooooooooooooooooooooooooo&@@@oooob
        dooooooooooooooooooooooooo&@@@ooooob
        dooooooooooooooooooooooooo&@@oooooob
        `dooooooooooooooooooooooooo&@ooooob'
         `doooooooooooooooooooooooooooooob'
          `doooooooooooooooooooooooooooob'
           `doooooooooooooooooooooooooob'
            `doooooooooooooooooooooooob'
             `doooooooooooooooooooooob'
              `dooooooooobodoooooooob'
               `doooooooob dooooooob'
                 `"""""""' `""""""'

        They say comparing Harris and SIFT keypoints is like comparing
        apples and oranges...ok they don't say that...sue me.
"""

import sys
import cv2 as cv
import numpy as np

def harris(image, sigma):
    """
    Harris measure computation from lecture
    Take in image and sigma
    Perform Gaussian smoothing, get the components of the outer product,
      convolution of outer product and Gaussian kernel, and compute
      Harris measure
    Return Harris measure
    """
    imgDx, imgDy = getDerivs(image, sigma)
    imgDxSq = imgDx * imgDx
    imgDySq = imgDy * imgDy
    imgDxDy = imgDx * imgDy
    hSigma = int(2*sigma)
    hKsize = (4*hSigma+1,4*hSigma+1)
    imgDxSq = cv.GaussianBlur(imgDxSq, hKsize, hSigma)
    imgDySq = cv.GaussianBlur(imgDySq, hKsize, hSigma)
    imgDxDy = cv.GaussianBlur(imgDxDy, hKsize, hSigma)
    kappa = 0.004
    imgDet = imgDxSq * imgDySq - imgDxDy * imgDxDy
    imgTrace = imgDxSq + imgDySq
    imgHarris = imgDet - kappa * imgTrace * imgTrace
    return imgHarris

def getDerivs(image, sigma):
    """
    Helper function for harris function
    Take in image and sigma
    Get gradient derivatives in x and y directions
    Return derivatives
    """
    ksize = (int(4*sigma+1),int(4*sigma+1))
    imgGauss = cv.GaussianBlur(image.astype(np.float64), ksize, sigma)
    kx,ky = cv.getDerivKernels(1,1,3)
    kx = np.transpose(kx/2)
    ky = ky/2
    imgDx = cv.filter2D(imgGauss,-1,kx)
    imgDy = cv.filter2D(imgGauss,-1,ky)
    return imgDx, imgDy

def nonMaxSuppression(image, sigma):
    """
    Non-max suppression from class without thresholding
    Normalize Harris measure, dilate based on maxDist(from Sigma),
      compare the two to get peaks
    Normalize peaks and return peaks
    """
    imgHarris = normalize(image)
    maxDist = int(2*sigma)
    kernel = np.ones((2*maxDist+1, 2*maxDist+1), np.uint8)
    imgHarris = imgHarris.astype(np.uint8)
    imgHarrisDilate = cv.dilate(imgHarris, kernel)
    imgPeaks = cv.compare(imgHarris, imgHarrisDilate, cv.CMP_GE)
    imgPeaks =  imgPeaks * image
    imgPeaks = normalize(imgPeaks)
    return imgPeaks

def normalize(image):
    """
    Helper function
    Takes in image
    Normalizes(scales) image to 255 (whites get whiter)
    Return normalized image
    """
    min = np.min(image)
    max = np.max(image)
    normalImg = 255*(image - min) / (max - min)
    return normalImg

def getHarrisKeyPoints(orgImage, imgPeaks, sigma):
    """
    Go through all the keypoints in the peaks and:
      Get coordinates
      Create new OpenCV keypoint with coordinates, sigma, and the magnitude
        of the peak (response)
      Add new keypoint to list of keypoints
      Add coordinates to lists of coordinates (helpful later)
    """
    sortHarris = np.sort(imgPeaks,axis=None)[::-1]
    keyPoints = list()
    kpXCoords = []
    kpYCoords = []
    # Since we only need 200 top keypoints, start with 500 since top 200
    #   might contain duplicates
    for i in range(500):
        x,y = np.where(imgPeaks==sortHarris[i])
        x,y = x.astype(int), y.astype(int)
        newKp = cv.KeyPoint(y, x, 2*sigma, -1, imgPeaks[x,y])
        keyPoints.append(newKp)
        kpXCoords.append(int(x))
        kpYCoords.append(int(y))

    """
    Go through the KeyPoints and filter out ones that are too close to
      each other
    """
    newKP = []
    newKPX = []
    newKPY = []
    for i in range(1, len(keyPoints)):
        xPrev, yPrev = keyPoints[i-1].pt
        x,y = keyPoints[i].pt
        if x-1 <= xPrev <= x+1 and y-1 <= yPrev <= y+1:
            pass
        else:
            newKP.append(keyPoints[i-1])
            newKPX.append(xPrev)
            newKPY.append(yPrev)

    """
    Return the top 200 keypoints along with their coordinates
    """
    return newKP[:200], (newKPY[:200], newKPX[:200])

def getSiftKeyPoints(image, sigma):
    """
    SIFT keypoint finder from lecture
    Take in image and sigma
    Get SIFT keypoints, filter out duplicates and ones with too high of a size
    Keep track of the coordinates of each keypoints
    Return top 200 keypoints along with their coordinates
    """
    sift = cv.xfeatures2d.SIFT_create()
    keyPoints = sift.detect(image, None)
    keyPoints.sort(key = lambda k: k.response)
    keyPoints = keyPoints[::-1]

    startInd = 0
    for i in range(len(keyPoints)):
        if (keyPoints[i].size > 3*sigma):
            startInd = i+1
        else:
            break

    kpUnique = [keyPoints[startInd]]
    x,y = keyPoints[startInd].pt
    siftXCoords = [float(y)]
    siftYCoords = [float(x)]
    for k in keyPoints[startInd+1:]:
        if (k.pt != kpUnique[-1].pt) and (k.size < 3*sigma):
            kpUnique.append(k)
            x,y = k.pt
            siftXCoords.append(float(y))
            siftYCoords.append(float(x))

    return kpUnique[:200], (siftXCoords, siftYCoords)

def outputStats(harrisKP, siftKP):
    """
    Take in Harris and SIFT keypoints and output stats about the top
      ten keypoints with the largest response
    """
    print("\nTop 10 Harris keypoints:")
    for i in range(10):
        x,y = harrisKP[i].pt
        response = harrisKP[i].response
        size = harrisKP[i].size
        print("{}: ({:.2f}, {:.2f}) {:.4f} {:.2f}".format(i, x, y, response, size))

    print("\nTop 10 SIFT keypoints:")
    for i in range(10):
        x,y = siftKP[i].pt
        response = siftKP[i].response
        size = siftKP[i].size
        print("{}: ({:.2f}, {:.2f}) {:.4f} {:.2f}".format(i, x, y, response, size))

def compareKeyPoints(harrisCoords, siftCoords, harrisOrSift):
    """
    Takes in Harris and SIFT keypoints
    Compares the top 100 Harris keypoints to the top 200 SIFT keypoints
      or vice versa
    Uses numpy to get the distances from each keypoint to the keypoints in
      the other list
    Output average distances and how the ranks of each keypoint compares to its
      corresponding keypoint in the other list
    """
    harrisX, harrisY = harrisCoords
    siftX, siftY = siftCoords

    if harrisOrSift == "Harris":
        siftOrHarris = "SIFT"
        oneHundredX, oneHundredY = np.asarray(harrisX)[:100], np.asarray(harrisY)[:100]
        twoHundredX, twoHundredY = np.asarray(siftX)[:200], np.asarray(siftY)[:200]
    else:
        siftOrHarris = "Harris"
        oneHundredX, oneHundredY = np.asarray(siftX)[:100], np.asarray(siftY)[:100]
        twoHundredX, twoHundredY = np.asarray(harrisX)[:200], np.asarray(harrisY)[:200]

    minDists = []
    indexDiffs = []
    for i in range(100):
        distances = np.sqrt((twoHundredX - oneHundredX[i])**2 + (twoHundredY - oneHundredY[i])**2)
        minDists.append(np.min(distances))
        indexDiffs.append(np.abs(np.argmin(distances) - i))

    print("\n{} keypoint to {} distances:\nnum_from 100 num_to 200".format(siftOrHarris, harrisOrSift))
    print("Median distance: {:.1f}".format(np.median(minDists)))
    print("Average distance: {:.1f}".format(np.average(minDists)))
    print("Median index difference: {:.1f}".format(np.median(indexDiffs)))
    print("Average index difference: {:.1f}".format(np.average(indexDiffs)))

if __name__ == "__main__":
    """
    Handle command line arguments
    """
    if len(sys.argv) != 3:
        print("Correct usage: p2_compare.py sigma img")
        sys.exit()
    else:
        sig = sys.argv[1]
        inImgName = sys.argv[2]

    try:
        sig = float(sig)
    except ValueError:
        print("Sigma must be real number!")
        sys.exit()

    try:
        inImg = cv.imread(inImgName, 0)
    except AttributeError:
        print("{} does not exit or is not a valid image!".format(inImgName))
        sys.exit()

    """
    Harris keypoints
    """
    harrisImg = harris(inImg, sig)
    harrisImgSup = nonMaxSuppression(harrisImg, sig)
    harrisKP, hCoords = getHarrisKeyPoints(inImg, harrisImgSup, sig)
    harrisOut = cv.drawKeypoints(inImg.astype(np.uint8), harrisKP, None)

    """
    SIFT keypoints
    """
    siftKP, sCoords = getSiftKeyPoints(inImg, sig)
    siftOut = cv.drawKeypoints(inImg.astype(np.uint8), siftKP, None)

    """
    Output images with keypoints drawn on them
    """
    name, ext = inImgName.split(".")
    cv.imwrite("{}_harris.{}".format(name, ext), harrisOut)
    cv.imwrite("{}_sift.{}".format(name, ext), siftOut)

    """
    Output stats
    """
    outputStats(harrisKP, siftKP)
    compareKeyPoints(hCoords, sCoords, "Harris")
    compareKeyPoints(hCoords, sCoords, "SIFT")
