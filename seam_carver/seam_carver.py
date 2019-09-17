"""
Jacob Kaplan
seam_carver.py

Take an image as input and remove enough rows or columns to make the image
square using the seam-carving technique.

                                  .-'\
                                     \:. \
                                     |:.  \
                                     /::'  \
                                  __/:::.   \
                          _.-'-.'`  `'.-'`'._\-"`";.-'-,
                       .`;    :      :     :      :   : `.
                      / :     :      :                 :  \
                     /        :/\          :   /\ :     :  \
                    ;   :     /\ \   :     :  /\ \      :   ;
                   .    :    /  \ \          /  \ \          .
                   ;        /_)__\ \ :     :/_)__\ \    :    ;
                  ;         `-----`' : ,   :`-----`'          ;
                  |    :      :       / \         :     :     |
                  |                  / \ \ :            :     |
                  |    :      :     /___\ \:      :           |
                  |    :      :     `----`'       :           |
                  ;        |;-.,__   :     :   __.-'|   :     ;
                   ;    :  ||   \ \``/'---'\`\` /  ||        ;
                    .    :  \\   \_\/       \_\/   //   '   .
                     ;       \'._    /\     /\ _.-'/   :   ;
                      \   :   `._`'-/ /\._./ /\  .'  :    /
                       `\  :     `-.\/__\__\/_.;'   :   /`
                         `\  '   :   :        :   :  /`
                           `-`.__`        :   :__.'-`
                                 `-..`.__.'..-`

                            Oh no! Not more carving!
"""
import sys
import cv2 as cv
import numpy as np

def drawSeam(image):
    """
    Take in image and draw first seam in red
    Calculate energy matrix
    Find seam
    Iterate through seam and replace pixel colors with red on the image
    Return drawn-on image
    """
    drawSeamImg = image.copy()
    engMatrix = energyMatrix(drawSeamImg)
    seam = findSeam(engMatrix)
    for i in range(0, len(seam)):
        drawSeamImg[i, seam[i]] = (0, 0, 255)
        #print("Writing on image at {} {}".format(i, seam[i]))
    return drawSeamImg

def resize(image, orientation):
    """
    Take in an image and its orientation (irrelevant to computation but need it later)
    Iterate for the difference between col and row and:
        Calculate the energy matrix of the image
        Find its seam
        Remove the seam/resize the image
        Repeat until image is square
        If its the 1st, 2nd, or last iteration, output seam stats
    Return seam-carved square image
    """
    resizeImage = image.copy()
    row, col = resizeImage.shape[:2]
    for i in range(col - row):
        engMatrix = energyMatrix(resizeImage)
        seam = findSeam(engMatrix)
        resizeImage = removeSeam(resizeImage, seam)
        if i == 0 or i == 1 or i == (col - row - 1):
            seamStats(seam, i, engMatrix, orientation)
    return resizeImage

def energyMatrix(image):
    """
    Take in image
    Convert image to grayscale
    Take derivatives in x an y directions
    Add derivatives for the energy matrix and return energy matrix
    """
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    sobelX = cv.Sobel(gray, cv.CV_64F,1,0)
    sobelY = cv.Sobel(gray, cv.CV_64F,0,1)
    engMatrix = np.abs(sobelX) + np.abs(sobelY)
    return engMatrix

def findSeam(engMatrix):
    """
    Take in an energy matrix
    Create a new matrix called weights that is identical to energy matrix
    Iterate from the 1st row of the weights matrix to the last row and:
        Calculate the local minimums of the row above it
        Add those local minimums to the current row of the weight matrix
        e.g weights[10] = [13, 4, 20, 2, 2, 0]
            findLocalMins(weights[10]) = [10^5, 4, 2, 2, 0, 10^5]
            weights[11] = [4, 4, 16, 13, 1, 9]
            weights[11] += localMins = [10^5, 8, 18, 15, 3, 1, 10^5]
    After weight matrix is calculated, find the index of the smallest weight in
        the last row of the weight matrix
    Backtrack up through weight matrix, finding the least costly path, adding
        each index of that path to the seam
    Return seam
    """
    row, col = engMatrix.shape[:2]
    weights = engMatrix.copy()
    for i in range(1, row):
        w = findLocalMins(weights[i-1])
        weights[i] += w

    seam = list()
    c = np.argmin(weights[-1])
    for i in range(row-1, -1, -1):
        seam.insert(0,c)
        minW = np.argmin(weights[i, c-1:c+2])
        c += minW - 1

    return seam

def findLocalMins(engMatrixRow):
    """
    Take in a row of an energy matrix
    Shift row to left, center, and right
    Assign hight weights the leftmost and rightmost elements
    Calculate localMins of the left, center, and right rows
    Return local mins

    e.g.
    gradientRow = [1, 10, 0, 4, 3, 11, 0]
    left = [1, 10, 0, 4, 3]
    center = [10, 0, 4, 3, 11]
    right = [0, 4, 3, 11, 0]
    localMins = [10^5, 0, 0, 0, 3, 3, 10^5]
    """
    localMins = np.zeros_like(engMatrixRow)
    left = engMatrixRow[:-2].copy()
    center = engMatrixRow[1:-1].copy()
    right = engMatrixRow[2:].copy()
    left[0] = 10**5
    right[-1] = 10**5
    minLC = np.minimum(left,center)
    minLCR = np.minimum(minLC, right)

    localMins[0] = 10**5
    localMins[1:-1] = minLCR
    localMins[-1] = 10**5

    return localMins

def removeSeam(orgImage, seam):
    """
    Take in an image and seam
    Create a new image of equal dimensions of original image except with 1 less column
    Iterate through the rows of the image and add all the pixels from the left of
        the seam in the original image to the new image and all the pixels to the
        right of the seam in the original image to the new image
        This way all pixels from each row of the original image will be added to
        the new image except the ones along the seam
    Return the image with the seam removed
    """
    image = orgImage.copy()
    row, col = image.shape[:2]
    removed = np.zeros((row, col-1, 3), np.float32)

    for i in range(0, row):
        removed[i, 0:seam[i]] = image[i, 0:seam[i]]
        removed[i, seam[i]:-1] = image[i, seam[i]+1:-1]
    return removed

def seamStats(seam, num, engMatrix, orientation):
    """
    Take in seam, seam number, energy matrix, and image orientation
    Output stats about the seam based on the orientation of the image
    """
    print("\nPoints on seam {}:".format(num))
    print("{}".format(orientation))
    x1,y1 = len(seam)//2, seam[len(seam)//2]
    x2,y2 = len(seam)-1, seam[len(seam)-1]
    if orientation == "horizontal":
        print("{}, 0".format(seam[0]))
        print("{}, {}".format(y1, x1))
        print("{}, {}".format(y2, x2))
    else:
        print("0, {}".format(seam[0]))
        print("{}, {}".format(x1, y1))
        print("{}, {}".format(x2, y2))
    avgEnergy = 0
    for i in range(1, len(seam)):
        avgEnergy += engMatrix[i, seam[i-1]]
    avgEnergy /= len(seam)
    print("Energy of seam {}: {:.2f}".format(num, avgEnergy))

if __name__ == "__main__":
    """
    Handle command line arguments
    """
    if len(sys.argv) != 2:
        print("Correct usage: p2_seam_carve.py img")
        sys.exit()
    else:
        imgName = sys.argv[1]
    try:
        img = cv.imread(imgName)
    except AttributeError:
        print("{} not in a valid format! Must be .jpg!".format(imgName))
        sys.exit()

    """
    Images to draw seam on and resize and dimensions
    """
    imgToResize = img.copy()
    seamImage = img.copy()
    row, col = img.shape[:2]

    """
    Draw seam and resize based on dimensions of image
    """
    if (row > col):
        seamImage = np.fliplr(seamImage)
        seamImage = np.rot90(seamImage)

        imgToResize = np.fliplr(imgToResize)
        imgToResize = np.rot90(imgToResize)

        seamImage = drawSeam(seamImage)

        resizedImg = resize(imgToResize, "horizontal")
        resizedImg = np.rot90(resizedImg, 3)
        resizedImg = np.fliplr(resizedImg)
    else:
        seamImage = drawSeam(seamImage)
        resizedImg = resize(imgToResize, "vertical")

    """
    Write seam-drawn and resized images
    """
    name, ext = imgName.split(".")
    cv.imwrite("{}_seam.{}".format(name, ext), seamImage)
    cv.imwrite("{}_final.{}".format(name, ext), resizedImg)
