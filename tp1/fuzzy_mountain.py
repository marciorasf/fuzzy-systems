#  %% imports
import pandas as pd
import numpy as np
from functools import reduce


# %% declare functions
def fuzzyMountain(data, nClusters):
    xColumns = data.columns

    centroids = pd.DataFrame(columns=xColumns)

    densities = calculateDensities(data, 1)

    maxDensityPoint, maxDensity = getMaxDensityPointAndValue(data, densities)

    centroids = centroids.append(maxDensityPoint)

    for _ in range(nClusters-1):
        densities = updateDensities(
            data,
            densities,
            maxDensityPoint,
            maxDensity,
            1.5
        )
        maxDensityPoint, maxDensity = getMaxDensityPointAndValue(
            data,
            densities
        )

        if(maxDensityPoint.name in centroids.index):
            return centroids

        centroids = centroids.append(maxDensityPoint)

    return centroids


def calculateDensities(data, radius):
    denominator = (radius/2)**2
    nPoints = data.shape[0]

    densities = pd.DataFrame(
        np.zeros((nPoints, 1)),
        columns=["density"]
    )

    for row_label, row in data.iterrows():
        temp0 = (data.loc[:, :].to_numpy() - row.to_numpy())**2

        temp1 = euclNormMatrix(temp0)

        temp2 = temp1/denominator

        temp3 = np.exp(-temp2)

        density = temp3.sum()

        densities.loc[row_label, "density"] = density

    return densities


def getMaxDensityPointAndValue(data, densities):
    maxDensityIndex = densities["density"].idxmax()

    maxDensity = densities.loc[maxDensityIndex, "density"]
    point = data.iloc[maxDensityIndex]

    return [point, maxDensity]


def updateDensities(data, densities, maxDensityPoint, maxDensity, radius):
    denominator = (radius/2)**2

    for row_label, row in data.iterrows():
        density = np.exp(- ((euclNorm(row - maxDensityPoint)**2)/denominator))
        densities.loc[row_label, "density"] -= maxDensity*density

    return densities


def formatData(rawData):
    xDimension = rawData.shape[1]
    xColumns = [f'x{i}' for i in range(xDimension)]
    data = pd.DataFrame(rawData, columns=xColumns)
    return data


def euclNorm(arr):
    return np.linalg.norm(arr, ord=2)


def euclNormMatrix(matrix):
    result = []
    for row in matrix:
        result.append(np.linalg.norm(row, ord=2))

    return np.array(result)
