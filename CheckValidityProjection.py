import numpy as np


def checkValidityProjection(proj_points, imCols, imRows):


    newPointsX = proj_points[:, 0]
    newPointsY = proj_points[:, 1]

    inRow_One = 0 <= newPointsY
    inRow_Two = newPointsY < imCols - 1
    inRow = inRow_One * inRow_Two

    inCol_One = 0 <= newPointsX
    inCol_Two = newPointsX < imRows - 1
    inCol = inCol_One * inCol_Two
    totCond = inRow * inCol
    notTotCond = ~totCond
    newPointsX[notTotCond] = 0
    newPointsY[notTotCond] = 0
    # newPointsX = newPointsX[~(newPointsX == 0)]
    # newPointsY = newPointsY[~(newPointsY == 0)]
    xnew = newPointsX[:,]
    ynew = newPointsY[:,]
    xnew = np.around(xnew).T
    ynew = np.around(ynew).T
    xnew = xnew.astype(np.int64).T
    ynew = ynew.astype(np.int64).T

    return xnew, ynew