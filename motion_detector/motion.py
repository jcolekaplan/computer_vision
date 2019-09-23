"""
    Jacob Kaplan
    motion.py

    Given are two images, I0 and I1 , taken by a single camera while it is 
    either stationary or moving through a scene where objects in the scene may
    be moving. (a) estimate the apparent motion vectors at a sparse set of 
    points, (b) determine whether or not the camera is moving, (c) if it is
    moving, find the “focus of expansion” induced by the camera’s motion, (d)
    determine which points are moving independent of the camera’s movement, and 
    (e) cluster such points into coherent objects.

                                 /\
                                <  >
                                 \/
                                 /\
                                /  \
                               /++++\
                              /  ()  \
                              /      \
                             /~`~`~`~`\
                            /  ()  ()  \
                            /          \
                           /*&*&*&*&*&*&\
                          /  ()  ()  ()  \
                          /              \
                         /++++++++++++++++\
                        /  ()  ()  ()  ()  \
                        /                  \
                       /~`~`~`~`~`~`~`~`~`~`\
                      /  ()  ()  ()  ()  ()  \
                      /*&*&*&*&*&*&*&*&*&*&*&\
                     /                        \
                    /,.,.,.,.,.,.,.,.,.,.,.,.,.\
                               |   |
                              |`````|
                              \_____/
                 Deck the hall with no more Comp Vis!
"""

import sys
import cv2 as cv
from functions import *

if __name__ == "__main__":
    """
    Handle command line arguments
    """
    if len(sys.argv) != 3:
        print("Usage: {} imgOne imgTwo".format(sys.argv[0]))
        sys.exit()
    else:
        imgOneName = sys.argv[1]
        imgTwoName = sys.argv[2]
    
    """
    Read in images and check if they're valid
    """
    imgOne = cv.imread(imgOneName, 1)
    imgTwo = cv.imread(imgTwoName, 1)
 
    if imgOne is None:
        print("{} not found!".format(imgOneName))
        sys.exit()
    
    if imgTwo is None:
        print("{} not found!".format(imgTwoName))
        sys.exit()
    
    """
    If valid, get a sparse set of points and estimate motion vectors at them
    Draw the motion vectors and output the image
    """
    imgOnePts, imgTwoPts = calcOptFlow(imgOne, imgTwo)
    motionVecs = drawMotionVectors(imgTwo, imgOnePts, imgTwoPts)

    """
    Use the points to determine if the camera is moving
    """
    isMoving = isCameraMoving(imgOnePts)
    
    """
    If the camera is moving, estimate the focus of expansion
    Draw focus of expansion and output the image
    """
    if isMoving:
        foc = getFocusOfExpansion(imgOne, imgTwo)
        motionVecs = drawFocusOfExpansion(imgTwo, foc, imgOnePts)

    """
    Output image with motion vectors and possibly focus of expansion
    """    
    nameOne, ext = imgOneName.split(".")
    nameTwo, _ = imgTwoName.split(".")
    cv.imwrite("{}_{}_mv.{}".format(nameOne, nameTwo, ext), motionVecs)
    
    """
    Find the objects moving independent of camera motion
    Draw rectangles around them and output that image
    """
    contours = motionDetector(imgOne, imgTwo)
    motionDetec = drawMovingObjects(imgTwo, contours)
    cv.imwrite("{}_{}_md.{}".format(nameOne, nameTwo, ext), motionDetec)

