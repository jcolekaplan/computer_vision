"""
    Jacob Kaplan
    shape.py
    
    Purpose: Given a set of point coordinates in R^2 , compute and output the following:
    (a) The minimum and maximum x and y values.
    (b) The center of mass (average) x and y values.
    (c) The axis along which the data varies the least and the standard deviation of this variation.
    (d) The axis along which the data varies the most and the standard deviation of that variation.
    (e) The closet point form of the best fitting line (through the original data).
    (f) The implicit form of the line.
    (g) A decision about the shape that best describes the data
    (h) Output a MatPlotLib plot saved as an image containing a scatter plot of
        the points and of the center of mass

                         *     ,MMM8&&&.            *
                              MMMM88&&&&&    .
                             MMMM88&&&&&&&
                 *           MMM88&&&&&&&&
                             MMM88&&&&&&&&
                             'MMM88&&&&&&'
                               'MMM8&&&'      *
                      |\___/|
                      )     (             .              '
                     =\     /=
                       )===(       *
                      /     \
                      |     |
                     /       \
                     \       /
              _/\_/\_/\__  _/_/\_/\_/\_/\_/\_/\_/\_/\_/\_
              |  |  |  |( (  |  |  |  |  |  |  |  |  |  |
              |  |  |  | ) ) |  |  |  |  |  |  |  |  |  |
              |  |  |  |(_(  |  |  |  |  |  |  |  |  |  |
              |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
              |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
            Sure you can take the eigenvalue of some points, but can you
            ever find the eigenvalue of your soul?
"""
import sys
import numpy as np
from numpy import linalg as la
#import matplotlib.pyplot as plt

def xyValues(points):
    """
    Take in array of points
    Return x values and y values of the points, respectively
    """
    return points[:,0], points[:,1]

def eigen(points):
    """
    Take in array of points
    Center points around the center of mass (mean)
    Create new 2xN matrix with the centered x values as the first row and
        the centered y values as the second row
    Use new 2xN matrix to create a covariance matrix
    Get eigenvalues and eigenvectors of the covariance matrix
    Return eigenvalues and eigenvectors
    """
    N = points.shape[0]
    xVals, yVals = xyValues(points)
    xVals -= np.mean(xVals)
    yVals -= np.mean(yVals)
    stackPoints = np.stack((xVals,yVals))
    covarMatrix = np.cov(stackPoints)
    eigenvals, eigenvecs = la.eig(covarMatrix)
    eigenvals = np.sqrt(eigenvals - eigenvals/N)
    return eigenvals, eigenvecs

def getMinAxis(evals, evecs):
    """
    Take in eigenvalues and eigenvectors of the points
    Return the first eigenvector and the second eigenvalue (these correspond
        to the minimum axis of the points)
    """
    minAxis = evecs[0]
    sMin = evals[1]
    return minAxis, sMin

def getMaxAxis(evals, evecs):
    """
    Take in eigenvalues and eigenvectors of the points
    Return the second eigenvector and the first eigenvalue (these correspond
        to the maximum axis of the points)
    """
    maxAxis = evecs[1]
    sMax = evals[0]
    return maxAxis, sMax

def getClosestPoint(minAxis, xAvg, yAvg):
    """
    Take in the info for the minimum axis, and the averages of the x and y values
    Calculate rho and p
    Return rho and p
    """
    rho = minAxis[0] * yAvg + minAxis[1] * xAvg
    p = np.arccos(minAxis[1])
    return rho, p

def getShape(sMin, sMax, tau):
    """
    Take in sMin, sMax, and tau
    Determine best fit and return it
    """
    if sMin < (tau * sMax):
        return "line"
    else:
        return "ellipse"

def plot(xVals, yVals, comX, comY, a, b, c, outfig):
    """
    Take in x and y values, the average of x and y values, the line of best fit,
        and the name of the file to save plot
    Set axes, plot x and y values, plot line of best fit, plot center of mass
    """
    axes = plt.gca()
    axes.set_xlim([0,55])
    axes.set_ylim([0,55])
    plt.scatter(xVals, yVals)
    x = np.linspace(0,51,102)
    a = (a/b)
    c = -1*(c/b)
    y = c - a*x
    plt.plot(x, y,'-k')
    plt.plot(comX, comY, markersize=8, color="red")
    plt.savefig(outfig)

if __name__ == "__main__":
    """
    Handle command line arguments
    """
    if len(sys.argv) != 4:
        print("Correct usage: p3_shape points tau outfig")
        sys.exit()
    else:
        pointsFile = sys.argv[1]
        tau = sys.argv[2]
        outfig = sys.argv[3]

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
        tau = float(tau)
    except ValueError:
        print("Tau must be number!")
        sys.exit()

    """
    Calculate and output stats
    """
    xValues, yValues = xyValues(points)
    xCopy = np.copy(xValues)
    yCopy = np.copy(yValues)
    xAvg, yAvg = np.mean(xValues), np.mean(yValues)
    print("min: ({:.3f},{:.3f})".format(np.min(xValues), np.min(yValues)))
    print("max: ({:.3f},{:.3f})".format(np.max(xValues), np.max(yValues)))
    print("com: ({:.3f},{:.3f})".format(xAvg, yAvg))

    eigenvals, eigenvecs = eigen(points)
    minAxis, sMin = getMinAxis(eigenvals, eigenvecs)
    maxAxis, sMax = getMaxAxis(eigenvals, eigenvecs)
    print("min axis: ({:.3f},{:.3f}), sd {:.3f}".format(minAxis[1], minAxis[0], sMin))
    print("max axis: ({:.3f},{:.3f}), sd {:.3f}".format(maxAxis[1], maxAxis[0], sMax))

    rho, theta = getClosestPoint(minAxis, xAvg, yAvg)
    a,b,c = minAxis[1], minAxis[0], -1*rho
    print("closest point: rho {:.3f}, theta {:.3f}".format(rho, theta))
    print("implicit: a {:.3f}, b {:.3f}, c {:.3f}".format(a,b,c))
    print("best as {}".format(getShape(sMin, sMax, tau)))

    #plot(xCopy, yCopy, xAvg, yAvg, a, b, c, outfig)
