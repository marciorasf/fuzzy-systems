#  %% imports
from scipy.io import loadmat
import pandas as pd
import numpy as np
from functools import reduce
import plotly.express as px


# %% declare functions
def fuzzyMountain(data):
    xColumns = data.columns

    centroids = pd.DataFrame(columns=xColumns)

    densities = calculateDensities(data, 1)

    maxDensityPoint, maxDensity = getMaxDensityPointAndValue(densities)

    centroids = centroids.append(maxDensityPoint)

    densities = updateDensities(
        data, densities, maxDensityPoint, maxDensity, 1.5)

    maxDensityPoint, maxDensity = getMaxDensityPointAndValue(densities)

    centroids = centroids.append(maxDensityPoint)

    return centroids


def calculateDensities(data, radius):
    denominator = (radius/2)**2

    densities = pd.DataFrame(
        np.zeros((nPoints, 1)),
        columns=["density"]
    )

    for row_label, row in data.iterrows():
        density = reduce(
            lambda sum, x: sum +
            np.exp(- ((euclNorm(row - x)**2)/denominator)),
            data.loc[:, :].to_numpy(),
            0
        )
        densities.loc[row_label, "density"] = density

    return densities


def getMaxDensityPointAndValue(densities):
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


def formatData(data):
    xDimension = data.shape[1]
    xColumns = [f'x{i}' for i in range(xDimension)]
    data = pd.DataFrame(rawData, columns=xColumns)
    return data


def euclNorm(arr):
    return np.linalg.norm(arr, ord=2)


# %% main script
nPoints = 10
xDimension = 10
rawData = np.random.normal(0.5, 0.4, (nPoints, xDimension))

data = formatData(rawData)

centroids = fuzzyMountain(data)
