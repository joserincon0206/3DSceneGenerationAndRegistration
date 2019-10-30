import cv2
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from MyProjectPoints import myProjectPoints
from Pushforward import pushforward
from Pushforward import createRotationMatrix
from WarpImage import warpImage
from CheckValidityProjection import checkValidityProjection
from DeleleteRepeatedProjections import deleteRepeatedProjections
from scipy import interpolate
M = np.load('MsaterIntrinsic_mtx_3.npy')

def generateFakeData():

    pointCloud, topsEllipsoids, xv, yv, depth, color = generateInitialPointCloud()

    angleRotation = 180
    tVect = np.array([xv[300, 400] + 0., yv[300, 400] + 0., 13])

    point_cloud_matrix_Camera1_Frame = rigidDeformFromWorldToCamera(pointCloud, angleRotation, tVect)

    topsEllipsoids_Camera1_Frame = rigidDeformFromWorldToCamera(topsEllipsoids, angleRotation, tVect)

    projected_points = myProjectPoints(point_cloud_matrix_Camera1_Frame.T, M)

    projected_tops_of_ellipsoids_tr = myProjectPoints(topsEllipsoids_Camera1_Frame.T, M)
    projected_points_X = np.round(projected_points[:, 0])
    projected_points_Y = np.round(projected_points[:, 1])
    projected_points_X_isInRange = cv2.inRange(projected_points_X, 0, np.shape(xv)[1] - 1)
    projected_points_Y_isInRange = cv2.inRange(projected_points_Y, 0, np.shape(xv)[0] - 1)
    projected_points_X_isInRange = projected_points_X_isInRange == 0
    projected_points_Y_isInRange = projected_points_Y_isInRange == 0
    projected_points_X_isInRange = np.reshape(projected_points_X_isInRange, np.size(projected_points_X_isInRange), )
    projected_points_Y_isInRange = np.reshape(projected_points_Y_isInRange, np.size(projected_points_Y_isInRange), )

    projected_points_X = projected_points_X.astype(np.int)
    projected_points_Y = projected_points_Y.astype(np.int)
    projected_points_X[projected_points_X_isInRange] = 0
    projected_points_Y[projected_points_Y_isInRange] = 0

    newImage = np.zeros((720, 1280, 3))
    depth_flat = depth.flatten()
    color_1 = color[:, :, 0].flatten()
    color_2 = color[:, :, 1].flatten()
    color_3 = color[:, :, 2].flatten()

    newImage[projected_points_Y, projected_points_X, 0] = color_1
    newImage[projected_points_Y, projected_points_X, 1] = color_2
    newImage[projected_points_Y, projected_points_X, 2] = color_3

    for index in range(0, np.shape(projected_tops_of_ellipsoids_tr)[0]):
        point = projected_tops_of_ellipsoids_tr[index, :]
        if not np.isnan(point[0]) or not np.isnan(point[1]):
            newImage = cv2.circle(newImage, (int(point[0]), int(point[1])), 30, (0, 0, 255), -1)

    return point_cloud_matrix_Camera1_Frame, topsEllipsoids_Camera1_Frame, newImage, projected_points_Y, \
            projected_points_X, projected_tops_of_ellipsoids_tr, color


def generateInitialPointCloud():
    x_vector = np.arange(0., 6.4, 0.005)
    y_vector = np.arange(0., 3.6, 0.005)

    xv, yv = np.meshgrid(x_vector, y_vector)
    depth = np.zeros_like(xv)

    centers_spheres_x = [100, 300, 500, 700, 900]
    centers_spheres_y = [100, 300, 500]

    radius = 0.1

    for cx in centers_spheres_x:
        for cy in centers_spheres_y:
            a = np.power(radius, 2) - (np.power(xv - xv[cy, cx], 2) + np.power(yv - yv[cy, cx], 2) / 10)
            is_valid = a > 0
            z_chunk = np.zeros((np.shape(a)[0], np.shape(a)[1]))
            z_chunk[is_valid] = np.sqrt(a[is_valid])
            depth[cy - 100: cy + 100, cx - 100:cx + 100] = z_chunk[cy - 100: cy + 100, cx - 100:cx + 100]
    horizontal_increase = [200, 400, 600]
    is_not_zero = depth != 0

    depth[:, 0:200] = depth[:, 0:200] + 0
    depth[:, 200:400] = depth[:, 200:400] + 0.3
    depth[:, 400:600] = depth[:, 400:600] + 0.5
    depth[:, 600:800] = depth[:, 600:800] + 0.3

    depth[-300:-1, 0:200] = np.nan
    depth[~is_not_zero] = np.nan

    color = np.zeros((720, 1280, 3))

    color[:, 0:200, :] = (255, 0, 0)
    color[:, 200:400, :] = (0, 255, 0)
    color[:, 400:600, :] = (255, 255, 0)
    color[:, 600:800, :] = (255, 0, 0)
    color[:, 800:1280, :] = (255, 0, 255)

    tops_of_ellipsoids = np.zeros((4, 15))
    count = 0
    for cx in centers_spheres_x:
        for cy in centers_spheres_y:
            tops_of_ellipsoids[:, count] = np.array([xv[cy, cx], yv[cy, cx], depth[cy, cx], 1])
            count = count + 1

    depth[~is_not_zero] = 0
    is_nan_depth = np.isnan(depth)
    depth[is_nan_depth] = 0
    is_not_zero = depth != 0
    color[is_not_zero, :] = (255, 255, 255)

    point_cloud_matrix = np.zeros((4, np.size(xv)))
    point_cloud_matrix[0, :] = xv.flatten().T
    point_cloud_matrix[1, :] = yv.flatten().T
    point_cloud_matrix[2, :] = depth.flatten().T
    point_cloud_matrix[3, :] = 1

    return point_cloud_matrix, tops_of_ellipsoids, xv, yv, depth, color


def rigidDeformFromWorldToCamera(pointCloud, angleRotation, translation):
    Rx = np.array([[1, 0, 0], [0, np.cos(np.deg2rad(angleRotation)), -np.sin(np.deg2rad(angleRotation))],\
                   [0, np.sin(np.deg2rad(angleRotation)), np.cos(np.deg2rad(angleRotation))]])

    rigidTransformation = np.zeros((3, 4))
    rigidTransformation[:, 0:3] = np.eye(3)
    rigidTransformation[:, 3] = -translation
    point_cloud_matrix_transformed = np.dot(rigidTransformation, pointCloud)
    point_cloud_matrix_transformed2 = np.zeros((4, np.size(pointCloud[0, :])))
    point_cloud_matrix_transformed2[0:3, :] = point_cloud_matrix_transformed
    point_cloud_matrix_transformed2[3, :] = 1
    rigidTransformation = np.zeros((3, 4))
    rigidTransformation[:, 0:3] = Rx
    point_cloud_matrix_transformed = np.dot(rigidTransformation, point_cloud_matrix_transformed2)

    return point_cloud_matrix_transformed

def generateSecondCameraView(pointCloud, pointCloudTops, Rotation, Translation, numPoints, Intrinsics, Image, colorImage):

    imageSize = np.size(Image)/3
    point_cloud_matrix = np.zeros((4, np.size(Image)/3))
    point_cloud_matrix[0:3, :] = pointCloud
    point_cloud_matrix[3, :] = 1

    tops_of_ellipsoids_tr_m = np.zeros((4, 15))
    tops_of_ellipsoids_tr_m[0:3, :] = pointCloudTops
    tops_of_ellipsoids_tr_m[3, :] = 1


    imageCameraEllipse, tops_of_ellipsoids_tr_cm2, projected_tops_tr = pushforward(tops_of_ellipsoids_tr_m,
                                                                                   Rotation, Translation, 15, Intrinsics, Image,
                                                                                   colorImage)

    imageCamera2, point_cloud_matrix_cm2, projected_pointCloud = pushforward(point_cloud_matrix, Rotation, Translation, numPoints, Intrinsics, Image,
                                                             colorImage)

    for index in range(0, np.shape(projected_tops_tr)[0]):
        point = projected_tops_tr[index, :]
        if not np.isnan(point[0]) or not np.isnan(point[1]):
            imageCamera2 = cv2.circle(imageCamera2, (int(point[0]), int(point[1])), 10, (0, 0, 255), -1)

    return imageCamera2, projected_pointCloud, projected_tops_tr


def projectPointsOnSecondCameraView(pointCloudTops, Rotation, Translation, numPoints, Intrinsics, Image, colorImage):

    myImage = np.copy(Image)
    tops_of_ellipsoids_tr_m = np.zeros((4, np.size(pointCloudTops[0,:])))
    tops_of_ellipsoids_tr_m[0:3, :] = pointCloudTops
    tops_of_ellipsoids_tr_m[3, :] = 1

    imageCameraEllipse, tops_of_ellipsoids_tr_cm2, projected_tops_tr = pushforward(tops_of_ellipsoids_tr_m,
                                                                                   Rotation, Translation, np.size(pointCloudTops[0,:]),
                                                                                   Intrinsics, myImage,
                                                                                   colorImage)

    for index in range(0, np.shape(projected_tops_tr)[0]):
        point = projected_tops_tr[index, :]
        if not np.isnan(point[0]) or not np.isnan(point[1]):
            myImage = cv2.circle(myImage, (int(point[0]), int(point[1])), 10, (0, 0, 0), -1)

    return myImage, tops_of_ellipsoids_tr_cm2, projected_tops_tr



def getDerivativeOfRotationMatrix(x_angle, y_angle, z_angle):

    sin_x = np.sin(np.deg2rad(x_angle))
    cos_x = np.cos(np.deg2rad(x_angle))

    sin_y = np.sin(np.deg2rad(y_angle))
    cos_y = np.cos(np.deg2rad(y_angle))

    sin_z = np.sin(np.deg2rad(z_angle))
    cos_z = np.cos(np.deg2rad(z_angle))

    Rx = np.array([[1, 0, 0], ])


    Rx = np.array([[0, 0, 0], [0, -sin_x,  -cos_x], \
                   [0, cos_x, -sin_x]])
    Ry = np.array([[-sin_y, 0, cos_y], \
                   [0, 0, 0],
                   [-cos_y, 0, -sin_y]])
    Rz = np.array([[-sin_z, -cos_z, 0],\
                   [cos_z, -sin_z, 0],\
                   [0, 0, 0]])



    return Rx, Ry, Rz




if __name__ == '__main__':

    fakeData = generateFakeData()
    point_cloud_matrix_Camera1_Frame, topsEllipsoids_Camera1_Frame, ImageCamera1, projected_points_Y, projected_points_X, projected_tops_of_ellipsoids_tr, colorImage = fakeData
    RotationMatrix = createRotationMatrix(0, 0, 10)
    RotX, RotY, RotZ = (0, 0, 10)
    T = np.array([2, 0, 0])
    actualParameters = np.zeros(6, )
    actualParameters[0] = RotX
    actualParameters[1] = RotY
    actualParameters[2] = RotZ
    actualParameters[3:6] = T
    numPoints = np.size(ImageCamera1)/3
    ImageCamera2, projected_Camera2, projected_tops_of_ellipsoids_tr = generateSecondCameraView(point_cloud_matrix_Camera1_Frame, topsEllipsoids_Camera1_Frame, RotationMatrix, T, numPoints, M, ImageCamera1,
                             colorImage)
    cols_nan = np.isnan(topsEllipsoids_Camera1_Frame).any(axis=0)
    topsEllipsoids_Camera1_Frame = topsEllipsoids_Camera1_Frame[:, ~cols_nan]
    estimated_angle_x, estimated_angle_y, estimated_angle_z = (3, 5, 0)
    RotationMatrix = createRotationMatrix(0, 0, 12)
    estimated_T = np.array([2.5, 1.2, 0.99])
    estimatedParameters = np.zeros(6,)
    estimatedParameters[0] = estimated_angle_x
    estimatedParameters[1] = estimated_angle_y
    estimatedParameters[2] = estimated_angle_z
    estimatedParameters[3:6] = estimated_T
    actualData = np.reshape(projected_tops_of_ellipsoids_tr, (np.size(projected_tops_of_ellipsoids_tr), ))
    rows_nan = np.isnan(actualData)
    actualData = actualData[~rows_nan]
    writer = cv2.VideoWriter('Calibration.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 25, (1280, 720), True)
    for epochs in range(0, 800):

        estimated_angle_x = estimatedParameters[0]
        estimated_angle_y = estimatedParameters[1]
        estimated_angle_z = estimatedParameters[2]
        RotationMatrix = createRotationMatrix(estimated_angle_x, estimated_angle_y, estimated_angle_z)
        estimated_T = estimatedParameters[3:6]
        ImageEstimate, estimate_ptCld, estimate_projection = projectPointsOnSecondCameraView(
            topsEllipsoids_Camera1_Frame, RotationMatrix, estimated_T, numPoints, M, ImageCamera2, colorImage)
        ImageEstimate = cv2.normalize(src=ImageEstimate, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,
                                   dtype=cv2.CV_8UC1)
        estimate_projection = np.reshape(estimate_projection, np.size(estimate_projection))
        Rx, Ry, Rz = getDerivativeOfRotationMatrix(estimated_angle_x, estimated_angle_y, estimated_angle_z)
        error = actualData - estimate_projection
        errorT = np.sum(error)/np.size(error)
        bigJacobian = np.zeros((2*np.size(estimate_ptCld[0, :]), 6))
        for indexPoints in range(0, np.size(estimate_ptCld[0, :])):

            point_minus_translation = estimate_ptCld[:, indexPoints] - estimated_T

            Dx = np.dot(Rx, point_minus_translation)

            Dy = np.dot(Ry, point_minus_translation)

            Dz = np.dot(Rz, point_minus_translation)

            Dt = -RotationMatrix

            DRot = np.zeros((3, 3))

            DRot[:, 0] = Dx.T

            DRot[:, 1] = Dy.T

            DRot[:, 2] = Dz.T

            ux, uy, uz = estimate_ptCld[:, indexPoints]

            Ds = np.array([[1/uz, 0, -ux/(np.power(uz, 2))], [0, 1/uz, -uy/np.power(uz, 2)]])

            hello = 0

            Dh = M[0:2, 0:2]

            JacRot = np.dot(Ds, DRot)

            JacRot = np.dot(Dh, JacRot)

            JacTrans = np.dot(Ds, Dt)

            JacTrans = np.dot(Dh, JacTrans)

            JacTotal = np.concatenate((JacRot, JacTrans), axis=1)

            mod2Num = np.mod(indexPoints, 2)

            bigJacobian[2*indexPoints:2*indexPoints+2, :] = JacTotal

        newEstimate = np.dot(np.linalg.pinv(bigJacobian), error.T)

        estimatedParameters[0] = estimatedParameters[0] + 0.015 * np.rad2deg(newEstimate[0])

        estimatedParameters[1] = estimatedParameters[1] + 0.015 * np.rad2deg(newEstimate[1])

        estimatedParameters[2] = estimatedParameters[2] + 0.015 * np.rad2deg(newEstimate[2])

        estimatedParameters[3] = estimatedParameters[3] + 0.015 * newEstimate[3]

        estimatedParameters[4] = estimatedParameters[4] + 0.015 * newEstimate[4]

        estimatedParameters[5] = estimatedParameters[5] + 0.015 * newEstimate[5]


        textRx = "Est :" + str(round(estimatedParameters[0],2))
        textRy = "Est :" + str(round(estimatedParameters[1], 2))
        textRz = "Est :" + str(round(estimatedParameters[2], 2))
        textTx = "Est :" + str(round(estimatedParameters[3], 2))
        textTy = "Est :" + str(round(estimatedParameters[4], 2))
        textTz = "Est :" + str(round(estimatedParameters[5], 2))
        textRx2 = "Rx Act :" + str(round(actualParameters[0], 2))
        textRy2 = "Rx Act :" + str(round(actualParameters[1], 2))
        textRz2 = "Rx Act :" + str(round(actualParameters[2], 2))
        textTx2 = "Rx Act :" + str(round(actualParameters[3], 2))
        textTy2 = "Rx Act :" + str(round(actualParameters[4], 2))
        textTz2 = "Rx Act :" + str(round(actualParameters[5], 2))

        cv2.putText(ImageEstimate, textRx2, (900, 50),  cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))
        cv2.putText(ImageEstimate, textRy2, (900, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))
        cv2.putText(ImageEstimate, textRz2, (900, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))
        cv2.putText(ImageEstimate, textTx2, (900, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))
        cv2.putText(ImageEstimate, textTy2, (900, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))
        cv2.putText(ImageEstimate, textTz2, (900, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))

        cv2.putText(ImageEstimate, textRx, (1100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))
        cv2.putText(ImageEstimate, textRy, (1100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))
        cv2.putText(ImageEstimate, textRz, (1100, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))
        cv2.putText(ImageEstimate, textTx, (1100, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))
        cv2.putText(ImageEstimate, textTy, (1100, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))
        cv2.putText(ImageEstimate, textTz, (1100, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))

        errorInText = "Reproject Error: " + str(round(errorT, 2))


        isLess = (abs(errorT) < 10)
        if isLess:
            cv2.putText(ImageEstimate, errorInText, (900, 350), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))
        else:
            cv2.putText(ImageEstimate, errorInText, (900, 350), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))

        writer.write(ImageEstimate)
        cv2.imshow('ImageEstimate', ImageEstimate)
        cv2.waitKey(1)



    cv2.imshow('ImageEstimate', ImageEstimate)
    cv2.waitKey(0)

    cv2.destroyAllWindows()


    hey = 0
