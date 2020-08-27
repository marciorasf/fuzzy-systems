# %%
from scipy.io import loadmat
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from functools import reduce
from c_means import cMeans
import cv2
from rescale_image import getImage

# %%
def printDataAndCentroids3d(data, centroids, keys):
    fig = make_subplots(x_title="x", y_title="y")

    pointColors = []
    for _, row in data.iterrows():
        pointColors.append(f'rgb({row["x2"]}, {row["x1"]}, {row["x0"]})')

    # Add traces
    fig.add_trace(
        go.Scatter3d(
            x=data[keys[0]],
            y=data[keys[1]],
            z=data[keys[2]],
            mode="markers",
            name="Input points",
            opacity=0.5,
            marker=dict(color=pointColors, size=3),
        )
    )

    centroidColors = []
    for _, row in centroids.iterrows():
        centroidColors.append(f'rgb({row["x2"]}, {row["x1"]}, {row["x0"]})')

    fig.add_trace(
        go.Scatter3d(
            x=centroids[keys[0]],
            y=centroids[keys[1]],
            z=centroids[keys[2]],
            mode="markers",
            name="Centroids",
            marker=dict(color=centroidColors, size=15),
        )
    )
    fig.show()


def runCMeans(photo, scaleRatio, nClusters):
    image = getImage(f"./input/{photo}.jpg", scaleRatio)
    cv2.imwrite(f"./output/{photo}.jpg", image)

    flatImage = np.array([item for sublist in image for item in sublist])
    data, centroids, _ = cMeans(flatImage, nClusters, 1e-6, 1000, 2)

    data.to_csv(f"output/data_{photo}.csv", compression=None)
    centroids.to_csv(f"output/centroids_{photo}.csv", compression=None)
    printDataAndCentroids3d(data, centroids, ["x0", "x1", "x2"])


nClustersPerPhoto = {
    "photo001": 7,
    "photo002": 4,
    "photo003": 7,
    "photo004": 10,
    "photo005": 10,
    "photo006": 4,
    "photo007": 5,
    "photo008": 9,
    "photo009": 12,
    "photo010": 11,
    "photo011": 3,
}
scaleRatio = 0.5

for photo, nClusters in nClustersPerPhoto.items():
    try:
        runCMeans(photo, scaleRatio, nClusters)
    except:
        pass

