"""
Jacob Kaplan
camera.py

Construct a camera matrix and apply it to project points onto an image plane.
                             ___
                            / _ \
                           | / \ |
                           | \_/ |
                            \___/ ___
                            _|_|_/[_]\__==_
                           [---------------]
                           | O   /---\     |
                           |    |     |    |
                           |     \___/     |
                           [---------------]
                                 [___]
                                  | |\\
                                  | | \\
                                  [ ]  \\_
                                 /|_|\  ( \
                                //| |\\  \ \
                               // | | \\  \ \
                              //  |_|  \\  \_\
                             //   | |   \\
                            //\   | |   /\\
                           //  \  | |  /  \\
                          //    \ | | /    \\
                         //      \|_|/      \\
                        //        [_]        \\
                       //          H          \\
                      //           H           \\
                     //            H            \\
                    //             H             \\
                   //              H              \\
                  //                               \\
                 //                                 \\

                        Lights...camera...Comp Vis!
"""

import sys
import numpy as np
from numpy import sin, cos

def getIntrinsic(f, d, ic, jc):
    """
    Get intrinsic camera matrix, K, from the camera's focal length (f), pixel
        dimensions (d), and optical axis center (ic, jc)
    Convert pixel dimensions to millimeters by dividing by 1,000
    Get the adjusted focal length s_f by dividing f by d
    Construct and return K
    """
    d /= 1000
    s = f / d
    K = np.asmatrix([[s, 0, ic],
                     [0, s, jc],
                     [0, 0, 1]])
    return K

def getExtrinsic(rotVec, transVec):
    """
    Get extrinsic camera matrix, R_t, from the rotation and translation vectors
        of the camera
    Convert rotational vector to radians
    Construct the x, y, and z components of the camera's rotation matrix
    Multiply the x, y, and z componets to get the camera's rotation matrix, R
        Concatenate the transposed rotation matrix and translation matrix
        (transposed translation vector) multiplied by the transposed rotation
        matrix and -1
    Compute the center of the camera and it's axis direction
    Return R_t, camera center, and axis direction
    """
    rx, ry, rz = (np.pi * rotVec) / 180
    Rx = np.asmatrix([[1, 0,       0         ],
                      [0, cos(rx), -1*sin(rx)],
                      [0, sin(rx), cos(rx)   ]])

    Ry = np.asmatrix([[cos(ry),    0, sin(ry)],
                      [0,          1, 0      ],
                      [-1*sin(ry), 0, cos(ry)]])

    Rz = np.asmatrix([[cos(rz), -1*sin(rz), 0],
                      [sin(rz), cos(rz),    0],
                      [0,       0,          1]])

    R = np.matmul(Rx, Ry)
    R = np.matmul(R, Rz)
    t = transVec.transpose()
    Rt = np.hstack((R.transpose(), -1*R.transpose()*t))

    E = np.asmatrix([0,0,1]).transpose()
    cameraAxis = np.matmul(R.transpose(),E)
    cameraCenter = -1*R.transpose()*t

    return Rt, cameraCenter, cameraAxis

def output(P, points, cameraCenter, cameraAxis):
    """
    Output stats of the camera matrix, P, and projections of points using P
    Print P
    Iterate throught the points, get their 2D projections, determine if they
        are inside the boundaries of the image, and output results
    Determine if the point is behind or in front of the camera and count the
        number of visible and hidden projections, output results
    """
    print("Matrix M:")
    for line in P:
        w,x,y,z = line[0,0], line[0,1], line[0,2], line[0,3]
        print("{:.2f}, {:.2f}, {:.2f}, {:.2f}".format(w,x,y,z))

    print("Projections:")
    axisInd, axisDirection = axisInfo(cameraAxis, cameraCenter)
    num = 0
    visible = ""
    hidden = ""
    for coord in points:
        x3D, y3D, z3D = coord
        coord2D = get2D(P, coord)
        y2D, x2D = coord2D
        inOut = inOrOut(coord2D)
        print("{}: {} {} {} => {:.1f} {:.1f} {}"
              .format(num, x3D, y3D, z3D, x2D, y2D, inOut))
        if axisDirection*coord[axisInd] >= cameraCenter[axisInd]:
            hidden += " " + str(num)
        else:
            visible += " " + str(num)
        num += 1

    print("visible:{}".format(visible))
    print("hidden:{}".format(hidden))

def axisInfo(camAxis, camCenter):
    """
    Get the axis info to determine if a point is in front of or behind camera
        using its center and axis of direction
    Get largest direction component from direction vector and return its index
        e.g. [-22, 14, 3] return 0
    If Z is positive, return 1 (since the camera is facing that Z direction)
        Else, return -1
    """
    min = np.argmin(camAxis)
    max = np.argmax(camAxis)
    if (np.abs(np.min(camAxis)) > np.max(camAxis)):
        axisInd = min
    else:
        axisInd = max
    if camCenter[2] > 0:
        axisDir = 1
    else:
        axisDir = -1
    return axisInd, axisDir

def get2D(P, coord):
    """
    Get 2D projection of a point by multiplying it by a camera matrix, P
    Format coord so it's a matrix
    Multiply coord and P
    Divide new coord by its z-component
    Return x and y components of 2D coord
    """
    coord = np.asmatrix(coord).transpose()
    coord = np.vstack((coord,1))
    multCoord = np.matmul(P, coord)
    coord2D = multCoord[0:2] / multCoord[2]
    return coord2D[0,0], coord2D[1,0]

def inOrOut(coord):
    """
    Take x,y coord and return "inside" if it's within the 6000x4000 image
        Else, return "outside"
    """
    x,y = coord
    if 0 <= x <= 6000 and 0 <= y <= 4000:
        return "inside"
    else:
        return "outside"

if __name__ == "__main__":
    """
    Handle command line arguments
    """
    if len(sys.argv) != 3:
        print("Correct usage: p1_camera.py paramsFile pointsFile")
        sys.exit()
    else:
        paramsFile = sys.argv[1]
        pointsFile = sys.argv[2]

    """
    Open and read paramsFile
    """
    try:
        openParams = open(paramsFile, "r")
    except FileNotFoundError:
        print("No file {} found".format(paramsFile))
        sys.exit()

    try:
        params = list(map(float, openParams.read().split()))
    except ValueError:
        print("Params must be a floats or ints!")
        sys.exit()

    """
    Convert file info to appropriate data types
    """
    rotationVec = np.asarray(params[0:3])
    translationVec = np.asmatrix(params[3:6])
    f, d, ic, jc = params[6:10]

    """
    Open and read pointsFile
    """
    try:
        openPoints = open(pointsFile, "r")
    except FileNotFoundError:
        print("No file {} found".format(pointsFile))
        sys.exit()

    try:
        points = np.loadtxt(openPoints, dtype=np.float64)
    except ValueError:
        print("Malformed points file: {}, must be numbers".format(pointsFile))
        sys.exit()

    """
    Calculate P
    """
    K = getIntrinsic(f, d, ic, jc)
    Rt, camCen, camAx = getExtrinsic(rotationVec, translationVec)
    P = K * Rt

    """
    Output stats
    """
    output(P, points, camCen, camAx)
