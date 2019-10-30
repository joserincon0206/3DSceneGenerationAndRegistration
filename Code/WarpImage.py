import numpy as np
import cv2
def warpImage(proj_points_infra_frame_IRL, SampledImage):


    newPointsX = proj_points_infra_frame_IRL[:, 0]
    newPointsY = proj_points_infra_frame_IRL[:, 1]

    inRow_One = 0 <= newPointsY
    inRow_Two = newPointsY < np.shape(SampledImage)[0]
    inRow = inRow_One * inRow_Two

    inCol_One = 0 <= newPointsX
    inCol_Two = newPointsX < np.shape(SampledImage)[1]
    inCol = inCol_One * inCol_Two
    totCond = inRow * inCol
    notTotCond = ~totCond
    #newPointsX = newPointsX[totCond]
    #newPointsY = newPointsY[totCond]
    newPointsX[notTotCond] = np.nan
    newPointsY[notTotCond] = np.nan
    xnew = newPointsX
    ynew = newPointsY
    # xnew= newPointsX
    # ynew = newPointsY
    SampledImage_r = SampledImage[:,:,0]
    SampledImage_g = SampledImage[:, :, 1]
    SampledImage_b = SampledImage[:, :, 2]
    xv_deformed_grid = np.reshape(xnew, (SampledImage.shape[0], SampledImage.shape[1]))
    yv_deformed_grid = np.reshape(ynew, (SampledImage.shape[0], SampledImage.shape[1]))
    dstMapX, dstMapY = cv2.convertMaps(xv_deformed_grid.astype(np.float32),
                                       yv_deformed_grid.astype(np.float32), cv2.CV_16SC2)
    mapped_img_r = cv2.remap(SampledImage_r, dstMapY, dstMapX, cv2.INTER_CUBIC)
    mapped_img_g = cv2.remap(SampledImage_g, dstMapY, dstMapX, cv2.INTER_CUBIC)
    mapped_img_b = cv2.remap(SampledImage_b, dstMapY, dstMapX, cv2.INTER_CUBIC)

    mapped_img = np.zeros_like(SampledImage)
    mapped_img[:,:,0] = mapped_img_r
    mapped_img[:, :, 1] = mapped_img_g
    mapped_img[:, :, 2] = mapped_img_b

    #mapped_img = cv2.cvtColor(mapped_img, cv2.COLOR_GRAY2RGB)
    return mapped_img, xv_deformed_grid, yv_deformed_grid