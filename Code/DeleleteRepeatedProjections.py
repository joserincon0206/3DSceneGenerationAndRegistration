import numpy as np

def deleteRepeatedProjections(pointCloud, projectedPoints):


    uniqueIndices, countUnique = np.unique(projectedPoints, axis=1, return_counts=True)
    duplicates = uniqueIndices[:, countUnique > 1]

    for index in range(0, np.shape(duplicates)[1]):
        duplicate_point = duplicates[:, index].T
        projectedPointsTrans = projectedPoints.T
        indexDuplicate = np.where((projectedPointsTrans == tuple(duplicate_point)).all(axis=1))
        indexDuplicate = np.asanyarray(indexDuplicate)
        points_of_interest = pointCloud[:, indexDuplicate.flatten()]
        points_of_interest = np.sqrt(np.sum(np.power(points_of_interest, 2), axis=0))
        minumum_value_index = np.argmin(points_of_interest)
        hey_bool = indexDuplicate == indexDuplicate[:, minumum_value_index]
        chunk = projectedPoints[:, indexDuplicate.flatten()]
        chunk[:, ~hey_bool.flatten()] = np.nan
        projectedPoints[:, indexDuplicate.flatten()] = chunk

    return projectedPoints
