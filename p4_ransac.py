"""
    Jacob Kaplan - kaplaj2
    CSCI 4270: Computational Vision
    Homework 2: Problem 4 - October 1, 2018
    p4_ransac

    Implement the RANSAC algorithm for fitting a line to a set of points.

                                    _
                                 ,:'/   _..._
                                // ( `""-.._.'
                                \| /    6\___
                                |     6      4
                                |            /
                                \_       .--'
                                (_'---'`)
                                / `'---`()
                              ,'        |
              ,            .'`          |
              )\       _.-'             ;
             / |    .'`   _            /
           /` /   .'       '.        , |
          /  /   /           \   ;   | |
          |  \  |            |  .|   | |
           \  `"|           /.-' |   | |
            '-..-\       _.;.._  |   |.;-.
                  \    <`.._  )) |  .;-. ))
                  (__.  `  ))-'  \_    ))'
                      `'--"`       `''"`
        I'll ransack those points for you...just like I ransacked your pillows...
        sorry about that by the way...
"""

import sys
import numpy as np

def RANSAC(points, samples, tau):
    """
    RANSAC implementation:
    Take in array of points, number of random samples to cycle through, and tau
    Initialize kMax, p, theta, and average inlier and outlier distances
    Iterate from 0 to samples:
        Generate two random indices
        If they don't equal each other:
            Get the two points the two indices correspond to
            Get the implicit form of the line between the two points
            Calculate theta and p
            Initialize k (# of inliers) and average inlier and outlier distance
            Check the number of inliers of the line
            If the number of inliers (k) is larger than kMax:
                Set new kMax, update average inlier and outlier distances
                Output RANSAC stats
    Output average inlier and outlier distance stats
    """
    N = points.shape[0]
    kMax = 0
    p = 0
    theta = 0
    avgInlierDists = list()
    avgOutlierDists = list()

    for m in range(samples):
        sample = np.random.randint(0, N, 2)
        i,j = sample

        if i != j:
            x1,y1 = points[i]
            x2,y2 = points[j]
            a,b,c = getImplicit(x1,y1,x2,y2)
            p = -1 * c
            theta = np.arctan2(b,a)
            k = 0
            avgInlierDist = 0
            avgOutlierDist = 0

            for n in range(N):
                xi,yi = points[n]
                if (xi*np.cos(theta) + yi*np.sin(theta) - p)**2 < tau**2:
                    avgInlierDist += np.abs(xi*np.cos(theta) + yi*np.sin(theta) - p)
                    k += 1
                else:
                    avgOutlierDist += np.abs(xi*np.cos(theta) + yi*np.sin(theta) - p)

            if k > kMax:
                kMax = k
                avgInlierDist /= k
                avgOutlierDist /= max(1,m-1)
                avgInlierDists.append(avgInlierDist)
                avgOutlierDists.append(avgOutlierDist)
                outputRANSAC(m, i, j, a, b, c, k)

    outputAvgDist(avgInlierDists[-1], avgOutlierDists[-1])

def getImplicit(x1, y1, x2, y2):
    """
    Take two points (x1,y1) and (x2,y2)
    Calculate and return the implicit form of the line between the two points
        in the form of the variables a,b,c where (ax+by+c=0)
    """
    a = y1-y2
    b = x2-x1
    c = (x1*y2) - (x2*y1)
    div = np.sqrt(a**2 + b**2)
    a /= div
    b /= div
    c /= div
    if a < 0 and b < 0:
        a *= -1
        b *= -1
        c *= -1
        
    return a,b,c

def outputRANSAC(sample, i, j, a, b, c, k):
    """
    Take in number of samples, the indices, the implit form of line, and k
    Output stats
    """
    print("Sample {}:".format(sample))
    print("indices ({},{})".format(i,j))
    print("line ({:.3f},{:.3f},{:.3f})".format(a,b,c))
    print("inliers {}\n".format(k))

def outputAvgDist(avgInlierDist, avgOutlierDist):
    """
    Take in the avergae inlier and outleir distances
    Output stats
    """
    print("avg inlier dist {:.3f}".format(avgInlierDist))
    print("avg outlier dist {:.3f}".format(avgOutlierDist))

if __name__ == "__main__":
    """
    Handle command line arguments
    """
    if len(sys.argv) != 4 and len(sys.argv) != 5:
        print("Correct usage: p4_ransac.py points samples tau [seed]")
        sys.exit()
    else:
        pointsFile = sys.argv[1]
        samples = sys.argv[2]
        tau = sys.argv[3]
        if len(sys.argv) == 5:
            seed = sys.argv[4]
            try:
                seed = int(seed)
                np.random.seed(seed)
            except ValueError:
                print("Seed must be an integer!")
                sys.exit()

    try:
        openFile = open(pointsFile, "r")
    except FileNotFoundError:
        print("No file {} found".format(pointsFile))
        sys.exit()

    try:
        points = np.loadtxt(openFile, dtype=np.float64)
    except ValueError:
        print("Malformed points file: {}, must be numbers".format(pointsFile))
        sys.exit()

    try:
        samples = int(samples)
    except ValueError:
        print("Samples value must be integer!")
        sys.exit()

    try:
        tau = float(tau)
    except ValueError:
        print("Tau must be a number!")
        sys.exit()

    RANSAC(points, samples, tau)
