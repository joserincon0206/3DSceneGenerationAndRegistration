import cv2
import numpy as np
from MyProjectPoints import myProjectPoints

def pushforward(pointCloud, R, T, imageSize, CameraIntrinsics, xv, color):

    rigidTransformation = np.zeros((3, 4))
    rigidTransformation[:, 0:3] = np.eye(3)
    rigidTransformation[:, 3] = -T.T
    point_cloud_transformed = np.dot(rigidTransformation, pointCloud)
    point_cloud_transformed2 = np.zeros((4, imageSize))
    point_cloud_transformed2[0:3, :] = point_cloud_transformed
    rigidTransformation = np.zeros((3, 4))
    rigidTransformation[:, 0:3] = R
    point_cloud_matrix_transformed = np.dot(rigidTransformation, point_cloud_transformed2)
    projected_points = myProjectPoints(point_cloud_matrix_transformed.T, CameraIntrinsics)

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
    color_1 = color[:, :, 0].flatten()
    color_2 = color[:, :, 1].flatten()
    color_3 = color[:, :, 2].flatten()

    if imageSize > 15:
        newImage[projected_points_Y, projected_points_X, 0] = color_1
        newImage[projected_points_Y, projected_points_X, 1] = color_2
        newImage[projected_points_Y, projected_points_X, 2] = color_3

    return newImage, point_cloud_matrix_transformed, projected_points


def createRotationMatrix(x_angle, y_angle, z_angle):
    Rx = np.array([[1, 0, 0], [0, np.cos(np.deg2rad(x_angle)), -np.sin(np.deg2rad(x_angle))], \
                   [0, np.sin(np.deg2rad(x_angle)), np.cos(np.deg2rad(x_angle))]])
    Ry = np.array([[np.cos(np.deg2rad(y_angle)), 0, np.sin(np.deg2rad(y_angle))], \
                   [0, 1, 0], [-np.sin(np.deg2rad(y_angle)), 0, np.cos(np.deg2rad(y_angle))]])
    Rz = np.array([[np.cos(np.deg2rad(z_angle)), -np.sin(np.deg2rad(z_angle)), 0], \
                   [np.sin(np.deg2rad(z_angle)), np.cos(np.deg2rad(z_angle)), 0], \
                   [0, 0, 1]])

    totalRotation = np.dot(Rx, Ry)
    totalRotation = np.dot(totalRotation, Rz)

    return  totalRotation

