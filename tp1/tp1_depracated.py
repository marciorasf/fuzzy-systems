# %%
from scipy.io import loadmat
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from functools import reduce

# %% Declare functions
def euclNorm(arr):
    return np.linalg.norm(arr, ord=2)

def printData():
    fig = px.scatter(data, x="x0", y="x1")
    fig.show()

def printCentroids():
    fig = px.scatter(centroids, x="x0", y="x1")
    fig.show()

def printDataAndCentroids():
    fig = make_subplots(x_title="x", y_title="y")

    # Add traces
    fig.add_trace(go.Scatter(
        x=data["x0"], 
        y=data["x1"], 
        mode="markers", 
        name="Input points"
        ))
    fig.add_trace(go.Scatter(
        x=centroids["x0"], 
        y=centroids["x1"], 
        mode="markers", 
        name="Centroids",
        marker=dict(
            color="red",
            size=15
        )
        ))
    fig.show()

# %% load data from file
rawData = loadmat("./fcm_dataset.mat")

# %% Initialize data
nClusters = 4
xDimension = rawData["x"].shape[1]
mParam = 2

xColumns = [f"x{i}" for i in range(xDimension)]
clusterColumns = [f"k{i}" for i in range(nClusters)]

data = pd.DataFrame(rawData["x"], columns=xColumns)
centroids = pd.DataFrame(
    np.zeros((nClusters, xDimension)), columns=xColumns, index=clusterColumns,
)

for clusterCol in clusterColumns:
    data[clusterCol] = 0

for row_label, row in data.iterrows():
    data.loc[row_label, clusterColumns] = np.random.dirichlet(
        np.ones(nClusters), size=1
    )[0]

for clusterCol in clusterColumns:
    centroids.loc[clusterCol, :] = [
        [np.dot((data[clusterCol] ** mParam), data[xCol]) for xCol in xColumns]
    ] / (data[clusterCol] ** 2).sum()


# %%
exponent = 2/(mParam-1)
for _ in range(10):
    for row_label, row in data.iterrows():
        x = row[xColumns]
        for clusterCol in clusterColumns:
            denominator = 0
            for centroid_label, centroid in centroids.iterrows():
                denominator += (euclNorm(x-centroids.loc[clusterCol, :])/euclNorm(x-centroid))**exponent
            row[clusterCol] = 1/denominator

    for clusterCol in clusterColumns:
        centroids.loc[clusterCol, :] = [
            [np.dot((data[clusterCol] ** mParam), data[xCol]) for xCol in xColumns]
        ] / (data[clusterCol] ** 2).sum()

printDataAndCentroids()

