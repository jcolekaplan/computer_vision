"""
Jacob Kaplan
edge_detection.py

Implement the non-maximum suppression step and then a thresholding step

        .     .       .  .   . .   .   . .    +  .
          .     .  :     .    .. :. .___---------___.
               .  .   .    .  :.:. _".^ .^ ^.  '.. :"-_. .
            .  :       .  .  .:../:            . .^  :.:\.
                .   . :: +. :.:/: .   .    .        . . .:\
         .  :    .     . _ :::/:               .  ^ .  . .:\
          .. . .   . - : :.:./.                        .  .:\
          .      .     . :..|:                    .  .  ^. .:|
            .       . : : ..||        .                . . !:|
          .     . . . ::. ::\(                           . :)/
         .   .     : . : .:.|. ######              .#######::|
          :.. .  :-  : .:  ::|.#######           ..########:|
         .  .  .  ..  .  .. :\ ########          :######## :/
          .        .+ :: : -.:\ ########       . ########.:/
            .  .+   . . . . :.:\. #######       #######..:/
              :: . . . . ::.:..:.\           .   .   ..:/
           .   .   .  .. :  -::::.\.       | |     . .:/
              .  :  .  .  .-:.":.::.\             ..:/
         .      -.   . . . .: .:::.:.\.           .:/
        .   .   .  :      : ....::_:..:\   ___.  :/
           .   .  .   .:. .. .  .: :.:.:\       :/
             +   .   .   : . ::. :.:. .:.|\  .:/|
             .         +   .  .  ...:: ..|  --.:|
        .      . . .   .  .  . ... :..:.."(  ..)"
         .   .       .      :  .   .: ::/  .  .::\
         You know what has no edge? Space....space dude...
"""

import sys
import cv2 as cv
import numpy as np

def getGradient(image, sigma):
    """
    Gradient magnitude and direction finder from class
    Takes in image and sigma, returns the magnitude of the gradient of the
      image based on sigma, and the directions(angles) of the magnitudes
    """
    kernelSize = (int(4*sigma+1), int(4*sigma+1))
    imgGauss = cv.GaussianBlur(image, kernelSize, sigma)
    kx,ky = cv.getDerivKernels(1,1,3)
    kx = np.transpose(kx/2)
    ky = ky/2
    imgDx = cv.filter2D(imgGauss,-1,kx)
    imgDy = cv.filter2D(imgGauss,-1,ky)
    imgGradientMag = np.sqrt(imgDx**2 + imgDy**2)
    imgGradientDir = np.arctan2(imgDx, imgDy)
    return imgGradientMag, imgGradientDir

def nonMaxSuppression(mag, dir):
    """
    Take in the magnitude and directions of a gradient
    Iterate the directions:
      Get corresponding color of the direction
      Assign neighbor directions to look ahead and behind pixel
      If the corresponding magnitude > 1:
        Assign that color to the direction/color image
      If the corresponding magnitude is a local maximum:
        Assign the magnitude to the non-max suppression image
    Count the number of local maximums before threshold and after, output results
    Return the non-max suppression image and direction/color image
    """
    M, N = mag.shape
    nonMax = np.zeros((M,N))
    colorImg = np.zeros((M,N,3))
    for i in range(1,M-1):
        for j in range(1,N-1):
                neighborDir, color = getNeighborDirection(dir[i,j])
                fx, fy = neighborDir
                if (mag[i,j] >= 1):
                    colorImg[i,j] = color

                if (mag[i,j] >= mag[i+fx, j+fy] and \
                    mag[i,j] >= mag[i-fx, j-fy] and \
                    mag[i,j] > 0):
                    nonMax[i,j] = mag[i,j]

    count = (nonMax>0).sum()
    print("Number after non-maximum: {}".format(count))
    afterOneThresh = (nonMax>1).sum()
    print("Number after 1.0 threshold: {}".format(afterOneThresh))
    return nonMax, colorImg

def getNeighborDirection(dir):
    """
    Takes in a direction(angle)
    Converts direction to degrees, if it's negative, add 360
    Check what color the direction corresponds to
    Return the corresponding neighbor directions and color
    """
    angle = np.rad2deg(dir)
    if (angle < 0):
        angle += 360

    # North-south
    if (0 <= angle < 22.5 or 157.5 <= angle < 202.5 or 337.5 <= angle <= 360):
        return (1,0), (255,0,0)
    # Northeast-southwest
    elif (112.5 <= angle < 157.5 or 292.5 <= angle < 337.5):
        return (1,-1), (255,255,255)
    # East-west
    elif (67.5 <= angle < 112.5 or 247.5 <= angle < 292.5):
        return (0,1), (0,0,255)
    # Northwest-southeast
    elif (22.5 <= angle < 67.5 or 202.5 <= angle < 247.5):
        return (1,1), (0,255,0)

def threshold(image, sigma):
    """
    Take in image and sigma
    Threshold image by setting all values in it less than 1 to zero
    Get the mean and standard deviation of remaining pixels
    Find best threshold value and threshold image using it
    Output the mean, standard deviation, best threshold, and number of pixels
      left after thresholding
    """
    image[image<1] = 0
    mean = np.mean(image[image>=1])
    std = np.std(image[image>=1])
    if (sigma == 0):
        thresh = mean + 0.5*std
    else:
        thresh = min(mean + 0.5*std, 30/sigma)
    image[image>thresh] = 255
    image[image<=thresh] = 0
    numAfterThresh = (image>0).sum()

    print("Average: {:.2f}".format(mean))
    print("Std dev: {:.2f}".format(std))
    print("Threshold: {:.2f}".format(thresh))
    print("Number after threshold: {}".format(numAfterThresh))
    return image

def normalize(image):
    """
    Take in image
    Normalize(scale) the image to 255 (whites becomes whiter)
    Return normalized image
    """
    min = np.min(image)
    max = np.max(image)
    normalImg = 255*(image - min) / (max - min)
    return normalImg

if __name__ == "__main__":
    """
    Handle command line arguments
    """
    if len(sys.argv) != 3:
        print("Correct usage: p1_edge.py sigma in_img")
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
        inImg = cv.imread(inImgName,0).astype(np.float64)
    except AttributeError:
        print("{} does not exit or is not a valid image!".format(inImgName))
        sys.exit()

    """
    Get gradient magnitudes and directions
    Use them in non-max suppression step to get the non-max and color image
    Normalize images and write them out
    """
    gradientMag, gradientDir = getGradient(inImg, sig)
    nonMaxImg, colorImg = nonMaxSuppression(gradientMag, gradientDir)
    threshImg = threshold(nonMaxImg, sig)
    gradientMag = normalize(gradientMag)
    threshImg = normalize(threshImg)

    name, ext = inImgName.split(".")
    cv.imwrite("{}_grd.{}".format(name, ext), gradientMag)
    cv.imwrite("{}_dir.{}".format(name, ext), colorImg)
    cv.imwrite("{}_thr.{}".format(name, ext), threshImg)
