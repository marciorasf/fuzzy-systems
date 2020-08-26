from scipy.io import loadmat
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from functools import reduce


def euclNorm(arr):
    return np.linalg.norm(arr, ord=2)


def cMeans(data, nClusters, mParam=2, maxIterations=False):
    xDimension = data.shape[1]

    xColumns = [f"x{i}" for i in range(xDimension)]
    clusterColumns = [f"k{i}" for i in range(nClusters)]

    dfData = pd.DataFrame(data, columns=xColumns)
    dfCentroids = pd.DataFrame(
        np.zeros((nClusters, xDimension)), columns=xColumns, index=clusterColumns,
    )

    for clusterCol in clusterColumns:
        dfData[clusterCol] = 0

    for row_label, row in dfData.iterrows():
        dfData.loc[row_label, clusterColumns] = np.random.dirichlet(
            np.ones(nClusters), size=1
        )[0]

    for clusterCol in clusterColumns:
        dfCentroids.loc[clusterCol, :] = [
            [np.dot((dfData[clusterCol] ** mParam), dfData[xCol]) for xCol in xColumns]
        ] / (dfData[clusterCol] ** 2).sum()

    exponent = 2 / (mParam - 1)
    for gen in range(2):
        for row_label, row in dfData.iterrows():
            x = row[xColumns]
            for clusterCol in clusterColumns:
                denominator = 0
                for _, centroid in dfCentroids.iterrows():
                    denominator += (
                        euclNorm(x - dfCentroids.loc[clusterCol, :])
                        / euclNorm(x - centroid)
                    ) ** exponent
                row[clusterCol] = 1 / denominator

        for clusterCol in clusterColumns:
            dfCentroids.loc[clusterCol, :] = [
                [
                    np.dot((dfData[clusterCol] ** mParam), dfData[xCol])
                    for xCol in xColumns
                ]
            ] / (dfData[clusterCol] ** 2).sum()

        print(f"iteration {gen}")

    return [dfData, dfCentroids]
