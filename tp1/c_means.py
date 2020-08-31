from scipy.io import loadmat
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from functools import reduce
import copy
from fuzzy_mountain import formatData, fuzzyMountain


def euclNorm(arr):
    return np.linalg.norm(arr, ord=2)


def removeZeros(arr):
    for index, value in enumerate(arr):
        if(value == 0):
            arr[index] = 1e-15

    return arr


def cMeans(data, nClusters, tolerance=1e-15, maxIterations=2, mParam=2):
    xDimension = data.shape[1]
    nRows = data.shape[0]
    xColumns = [f"x{i}" for i in range(xDimension)]
    clusterColumns = [f"k{i}" for i in range(nClusters)]

    dfData = pd.DataFrame(data, columns=xColumns)

    mountainCentroids = fuzzyMountain(data, nClusters)
    dfCentroids = pd.DataFrame(
        mountainCentroids.to_numpy(), columns=xColumns, index=clusterColumns,
    )

    for clusterCol in clusterColumns:
        dfData[clusterCol] = 0

    dfData.loc[:, clusterColumns] = np.random.dirichlet(
        np.ones(nClusters), size=(nRows)
    )

    exponent = 2 / (mParam - 1)
    for iterations in range(maxIterations):
        for clusterCol in clusterColumns:
            X = dfData.loc[:, xColumns]
            denominator = np.zeros((nRows, 1))

            for _, centroid in dfCentroids.iterrows():
                tempNum = np.array(
                    list(
                        map(
                            euclNorm,
                            np.array(
                                (X - dfCentroids.loc[clusterCol, xColumns])),
                        )
                    )
                ).reshape((nRows, 1))

                tempDen = (
                    np.array(list(map(euclNorm, np.array((X - centroid)))))
                ).reshape((nRows, 1))

                tempDen = removeZeros(tempDen)

                denominator += (tempNum / tempDen) ** exponent

                denominator = removeZeros(denominator)

            dfData.loc[:, clusterCol] = np.ones((nRows, 1)) / denominator

        previousCentroids = copy.deepcopy(dfCentroids)

        for clusterCol in clusterColumns:
            dfCentroids.loc[clusterCol, :] = [
                [
                    np.dot((dfData[clusterCol] ** mParam), dfData[xCol])
                    for xCol in xColumns
                ]
            ] / (dfData[clusterCol] ** mParam).sum()

        delta = np.array(
            list(
                map(
                    euclNorm,
                    np.array(
                        (
                            dfCentroids.loc[:, xColumns]
                            - previousCentroids.loc[:, xColumns]
                        )
                    ),
                )
            )
        ).mean()

        if delta < tolerance:
            break

    return [dfData, dfCentroids, iterations]
