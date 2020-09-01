#  %% imports
import pandas as pd
import numpy as np
from functools import reduce


# %% declare functions
def fuzzyMountain(data, nClusters, radiusA):
    data = formatData(data)

    xColumns = data.columns
    normalizedData = normalize(data)
    centroids = pd.DataFrame(columns=xColumns)

    densities = calculateDensities(data, radiusA)

    maxDensityPoint, maxDensity = getMaxDensityPointAndValue(
        normalizedData, densities)

    centroids = centroids.append(maxDensityPoint)

    radiusB = 1.5*radiusA
    for _ in range(nClusters-1):
        densities = updateDensities(
            normalizedData,
            densities,
            maxDensityPoint,
            maxDensity,
            radiusB
        )
        maxDensityPoint, maxDensity = getMaxDensityPointAndValue(
            normalizedData,
            densities
        )

        centroids = centroids.append(maxDensityPoint)

    originalCentroids = centroids.copy()

    for rowLabel, _ in originalCentroids.iterrows():
        originalCentroids.loc[rowLabel] = data.loc[rowLabel]

    return originalCentroids


def normalize(data):
    normalizedData = data.copy(deep=True)

    for col in data:
        minVal = data[col].min()
        maxVal = data[col].max()

        delta = maxVal-minVal

        normalizedData[col] = (normalizedData[col].to_numpy()-minVal)/delta

    return normalizedData


def calculateDensities(data, radius):
    denominator = (radius/2)**2
    nPoints = data.shape[0]

    densities = pd.DataFrame(
        np.zeros((nPoints, 1)),
        columns=["density"]
    )

    for row_label, row in data.iterrows():
        density = np.exp(-(euclNormMatrix(
            (data.loc[:, :] - row)**2)/denominator)).sum()

        densities.loc[row_label, "density"] = density

    return densities


def getMaxDensityPointAndValue(data, densities):
    maxDensityIndex = densities["density"].idxmax()

    maxDensity = densities.loc[maxDensityIndex, "density"]
    point = data.iloc[maxDensityIndex]

    return [point, maxDensity]


def updateDensities(data, densities, maxDensityPoint, maxDensity, radius):
    densities = densities.copy()

    denominator = (radius/2)**2

    for row_label, row in data.iterrows():
        density = np.exp(- ((euclNorm(row - maxDensityPoint)**2)/denominator))
        densities.loc[row_label, "density"] -= maxDensity*density

    return densities


def formatData(rawData):
    xDimension = rawData.shape[1]
    xColumns = [f'x{i}' for i in range(xDimension)]
    data = pd.DataFrame(rawData, columns=xColumns, copy=True)
    return data


def euclNorm(arr):
    return np.linalg.norm(arr, ord=2)


def euclNormMatrix(matrix):
    matrix = np.array(matrix)
    result = np.zeros(matrix.shape[0])

    for row in matrix.T:
        result += np.sqrt(row**2)

    return np.array(result.T)
