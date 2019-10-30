import cv2
import numpy as np

def myProjectPoints(depth_array_padded, M2):

    shape_depth_array = np.shape(depth_array_padded)
    projPoints = np.zeros((shape_depth_array[0], 2))
    fx = M2[0, 0]
    fy = M2[1, 1]
    cx = M2[0, 2]
    cy = M2[1, 2]

    x = depth_array_padded[:, 0]
    y = depth_array_padded[:, 1]
    z = depth_array_padded[:, 2]
    x = x/z
    y = y/z
    u = fx*x + cx
    v = fy*y + cy
    projPoints[:,0] = u
    projPoints[:,1] = v
    return projPoints



