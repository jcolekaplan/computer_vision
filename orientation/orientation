"""
Jacob Kaplan
orientation.py

Implement the keypoint gradient direction estimation technique based on the
Lecture 9 notes


              /\
             /**\
            /****\   /\
           /      \ /**\
          /  /\    /    \        /\    /\  /\      /\            /\/\/\  /\
         /  /  \  /      \      /  \/\/  \/  \  /\/  \/\  /\  /\/ / /  \/  \
        /  /    \/ /\     \    /    \ \  /    \/ /   /  \/  \/  \  /    \   \
       /  /      \/  \/\   \  /      \    /   /    \
    __/__/_______/___/__\___\__________________________________________________

    I had considerable trouble finding the peaks...here they are...let's go ski
"""

import sys
import cv2 as cv
import numpy as np

def getGradient(image, sigma):
    """
    Gradient magnitude calculation from lecture
    Takes in image and sigma
    Calculates gradient magnitudes and directions(angles) based on sigma
    Returns magnitude and directions
    """
    kernelSize = (int(4*sigma+1), int(4*sigma+1))
    imgGauss = cv.GaussianBlur(image, kernelSize, sigma)
    kx,ky = cv.getDerivKernels(1,1,3)
    kx = np.transpose(kx/2)
    ky = ky/2
    imgDx = cv.filter2D(imgGauss,-1,kx)
    imgDy = cv.filter2D(imgGauss,-1,ky)
    imgGradient = np.sqrt(imgDx**2 + imgDy**2)
    imgDir = np.arctan2(imgDy, imgDx)
    imgDir = 180*imgDir/np.pi
    return imgGradient, imgDir

def pixelNeighborhood(point, image, sigma):
    """
    Takes in a point, an image, and sigma
    Calculates the width from sigma and creates a pixel "neighborhood"
      with the point as the center
    Returns the neighborhood
    """
    width = int(8*sigma)//2
    x,y = point
    neighborhood = image[x-width:x+width+1, y-width:y+width+1]
    return neighborhood

def getWeights(mag, sigma):
    """
    Takes in gradient magnitudes and sigma
    Generates Gaussian kernel based on sigma
    Creates 2D Gaussian kernel from outer product of the 1D kernel
    Multiplies the magnitudes by the kernel to get the weights of the pixels
    """
    gaussian = cv.getGaussianKernel(int(8*sigma+1), 2*sigma)
    window = np.outer(gaussian,gaussian)
    weights = mag * window
    return weights

def assignBins(angles, weights):
    """
    Takes in sorted angles(directions) and sorted weights
    Goes through each bin center (i.e. -175, -165, etc.):
      Gets the sum of the weights that fall in that bin
      Adds the sum to a list that represented a histogram
      i.e. bins has 36 slots with each slot a sum corresponding to each bin center
    Return the histogram
    """
    binCenters = np.arange(-175, 185, 10)
    hist = []
    for center in binCenters:
        sum = getSum(angles, weights, center - 10, center + 10, center)
        if (center == 175):
            sum += getSum(angles, weights, -180, -175, -185)
        elif (center == -175):
            sum += getSum(angles, weights, 175, 180, 185)
        hist.append(sum)
    return hist

def getSum(angles, weights, start, end, center):
    """
    Take in sorted list of angles, sorted list of weights, a start index,
      an end index, and a center index
    Gets the distance each angle is to the center
    Calculates the percent of the weight that falls into that bin
    Gets the indices of angles that fall between start and end
    Get the weights at those indices (i.e. weights at those angles)
    Multiply weights times the percentages and sum the result
    Return the sum
    """
    dist = np.abs(angles[(angles>=start) & (angles<=end)] - center)
    percent = 1 - dist / 10
    rangeInd = np.where((angles>=start) & (angles<=end))

    sum = 0
    if len(rangeInd[0]) != 0:
        rangeStart = np.min(rangeInd)
        rangeEnd = np.max(rangeInd) + 1
        weightRange = weights[rangeStart:rangeEnd]
        sum = (weightRange * percent).sum()
    return sum

def smoothHistogram(hist):
    """
    Take in histogram(bins with sum in each)
    Iterates through and "smooths" the weight based on the weight of its
      neighbor bins
    Returns the smoothed histogram
    """
    smoothedHist = []
    for i in range(len(hist)):
        if i == 0:
            neighborWeights = hist[1] + hist[-1]
        elif i == len(hist) - 1:
            neighborWeights = hist[-2] + hist[0]
        else:
            neighborWeights = hist[i-1] + hist[i+1]
        smoothWeight = (hist[i] + (neighborWeights/2))/2
        smoothedHist.append(smoothWeight)
    return smoothedHist

def findPeaks(hist):
    """
    Take in histogram
    Go through each bin in the histogram and:
      Find local maximum and:
        Fit a parabola around the two neighbor bins and local max bin
        Calculate the critical point that produces the max of the parabola
          (critical point represents orientation, max is the peak)
        Add both to list of peaks
    Return sorted list of peaks
    """
    peaks = []
    offsets = []
    binRanges = np.arange(-175, 185, 10)
    max = np.max(hist)
    for i in range(len(hist)):
        if i == 0:
            left, right = -1, 1
        elif i == len(hist) - 1:
            left, right = -2, 0
        else:
            left, right = i-1, i+1

        if (hist[i] - hist[left]) >= (0.01*max) \
            and (hist[i] - hist[right]) >= (0.01*max):
            a = (hist[right] - 2*hist[i] + hist[left]) / 2
            b = (hist[right] - hist[left]) / 2
            c = hist[i]
            aDx = a*2
            bDx = -1*b
            #critical point
            x = bDx/aDx
            # max
            max = a*(x**2) + b*x + c
            offset = (x*10) + binRanges[i]
            peaks.append((max, offset))

    return sorted(peaks, reverse=True)

def output(point, num, histogram, smoothHistogram, peaks):
    """
    Take in the point being evaluated, the number of that point, the histogram
      calculated, the smoothed histogram, and the list of peaks
    Output histogram and smoothed histogram info
    Output peak info and strong orientation peak info
    """
    print("\n Point {}: ({},{})\nHistograms:".format(num, point[0], point[1]))
    binRanges = np.arange(-180, 190, 10)
    for i in range(36):
        br1, br2 = binRanges[i], binRanges[i+1]
        h, sh = histogram[i], smoothHistogram[i]
        print("[{},{}]: {:.2f} {:.2f}".format(br1, br2, h, sh))

    maxPeak = 0
    for i in range(len(peaks)):
        peak, offset = peaks[i]
        print("Peak {}: theta {:.1f}, value {:.2f}".format(i, offset, peak))
        if peak > maxPeak:
            maxPeak = peak
    print("Number of strong orientation peaks: {}".format(strongPeaks(maxPeak, peaks)))

def strongPeaks(max, peaks):
    """
    Take in max peak and other peaks
    Count how many have a strong orientation (i.e. are 80% or above of the max)
    Return the count
    """
    strongPeaks = 0
    for i in range(len(peaks)):
        peak = peaks[i][0]
        if peak >= 0.8*max:
            strongPeaks += 1
    return strongPeaks

if __name__ == "__main__":
    """
    Handle command line arguments
    """
    if len(sys.argv) != 4:
        print("Correct usage: p2_compare.py sigma img points")
        sys.exit()
    else:
        sig = sys.argv[1]
        inImgName = sys.argv[2]
        pointsFileName = sys.argv[3]

    try:
        sig = float(sig)
    except ValueError:
        print("Sigma must be real number!")
        sys.exit()

    try:
        inImg = cv.imread(inImgName,0).astype(np.float64)
    except AttributeError:
        print("{} is not a valid image!".format(inImgName))
        sys.exit()

    try:
        points = np.loadtxt(pointsFileName, dtype=np.uint16)
    except ValueError:
        print("Malformed points file: {}, must be numbers".format(pointsFile))
        sys.exit()

    """
    Iterate through all the points in the file:
      Get gradient magnitudes and directions of whole image
      Crop those magnitudes and directions down to neighborhood around point
      Sort the directions and the weights (making sure each element still
        corresponds to other element in the other list)
      Assign bins (make histogram)
      Smooth the histogram
      Get the peaks
      Output all the info
    """
    for i in range(len(points)):
        point = points[i]
        gradientMag, gradientDir = getGradient(inImg, sig)
        gradientMag = pixelNeighborhood(point, gradientMag, sig)
        gradientDir = pixelNeighborhood(point, gradientDir, sig)

        dirSort = np.sort(gradientDir, axis=None)
        dirInd = np.argsort(gradientDir, axis=None)

        weights = getWeights(gradientMag, sig).flatten()
        weights = weights[dirInd]

        bins = assignBins(dirSort, weights)
        smoothedBins = smoothHistogram(bins)
        peaks = findPeaks(smoothedBins)

        output(points[i], i, bins, smoothedBins, peaks)
