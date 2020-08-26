from scipy.io import loadmat
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from functools import reduce
import copy


def euclNorm(arr):
    return np.linalg.norm(arr, ord=2)


def cMeans(data, nClusters, tolerance=1e-15, maxIterations=2, mParam=2):
    xDimension = data.shape[1]
    nRows = data.shape[0]
    xColumns = [f"x{i}" for i in range(xDimension)]
    clusterColumns = [f"k{i}" for i in range(nClusters)]

    dfData = pd.DataFrame(data, columns=xColumns)
    dfCentroids = pd.DataFrame(
        np.zeros((nClusters, xDimension)), columns=xColumns, index=clusterColumns,
    )

    for clusterCol in clusterColumns:
        dfData[clusterCol] = 0

    for row_label, _ in dfData.iterrows():
        dfData.loc[row_label, clusterColumns] = np.random.dirichlet(
            np.ones(nClusters), size=1
        )[0]

    for clusterCol in clusterColumns:
        dfCentroids.loc[clusterCol, :] = [
            [np.dot((dfData[clusterCol] ** mParam), dfData[xCol]) for xCol in xColumns]
        ] / (dfData[clusterCol] ** 2).sum()

    exponent = 2 / (mParam - 1)
    for _ in range(maxIterations):
        for clusterCol in clusterColumns:
            X = dfData.loc[:, xColumns]
            denominator = np.zeros((nRows, 1))
            
            for _, centroid in dfCentroids.iterrows():
                tempNum = np.array(
                    list(
                        map(
                            euclNorm,
                            np.array((X - dfCentroids.loc[clusterCol, xColumns])),
                        )
                    )
                ).reshape((nRows, 1))

                tempDen = (
                    np.array(list(map(euclNorm, np.array((X - centroid)))))
                ).reshape((nRows, 1))

                denominator += (tempNum / tempDen) ** exponent
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
        ).sum()
        print(delta)

        if delta < tolerance:
            break

    return [dfData, dfCentroids]
