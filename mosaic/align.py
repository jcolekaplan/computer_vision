"""
    Jacob Kaplan
    align.py

    Given a folder of N images as input:
    -  determine which pairs of images show the same scene,
    -  among those that do show the same scene, determine which images overlap
        sufficiently that they can be combined into a mosaic (either because the
        surface is flat or because the images were taken by a camera that was
        approximately rotated in place), and
    - for each pair of images that can be combined into a mosaic, generate and
        output a mosaic of the two images.
"""

import os
import sys
from functions import *

if __name__ == "__main__":
    """
    Handle command line arguments
    """
    if len(sys.argv) != 2:
        print("Usage: {} in_dir".format(sys.argv[0]))
        sys.exit()
    else:
        directory = sys.argv[1]
        outdir = directory + "/output/"
        if not os.path.exists(outdir):
            os.makedirs(outdir)

    try:
        images = findImages(directory)
    except FileNotFoundError:
        print("Directory does not exist!")
        sys.exit()

    numImgs = len(images)
    if numImgs < 2:
        print("Image directory must have at least 2 images!")
        sys.exit()

    for i in range(numImgs):
        for j in range(i+1, numImgs):
            img1, imgName1 = images[i]
            img2, imgName2 = images[j]
            name1, ext = imgName1.split(".")
            name2, ext = imgName2.split(".")

            # 1. Extract and match keyPoints ===================================
            kps1, kps2 = matchKeyPoints(img1, img2)
            drawBM = drawMatches(img1, img2, kps1, kps2)
            cv.imwrite(outdir + "{}_{}_bm.{}".format(name1, name2, ext), drawBM)
            outputMatchInfo(kps1, name1, name2)


            # 2. If enough matches, get fundamental matrix =====================
            if checkMatches(kps1):
                inl1, inl2 = getFundamental(kps1, kps2)
                drawFM = drawMatches(img1, img2, inl1, inl2)
                cv.imwrite(outdir + "{}_{}_fm.{}" \
                    .format(name1, name2, ext), drawFM)
                outputFundamentalInfo(kps1, inl1)

                #3-4. If enough survived, get homography matrix ================
                if checkInliers(kps1, inl1):
                    hm1, hm2, hmMat = getHomography(inl1, inl2)
                    drawHM = drawMatches(img1, img2, hm1, hm2)
                    cv.imwrite(outdir + "{}_{}_hm.{}" \
                        .format(name1, name2, ext), drawHM)
                    outputHomographyInfo(inl1, hm1)

                    #5-6 If aligned, make mosaic ===============================
                    if checkAlignment(hm1, inl1):
                        mosaic = buildMosaic(img2, img1, hmMat)
                        #outputMosaic(mosaic, imgName1, imgName2, outDir)
                        cv.imwrite(outdir + "{}_{}_mosiac.{}"\
                            .format(name1, name2, ext), mosaic)
