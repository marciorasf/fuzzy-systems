# %% imports
import numpy as np
import pandas as pd
from image_utils import readImage, writeImage
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# %% declare functions


def plotDataAndCentroids3d(data, centroids, keys):
    fig = make_subplots(x_title="x", y_title="y")

    pointColors = []
    for _, row in data.iterrows():
        pointColors.append(
            f'rgb({row[keys[2]]}, {row[keys[1]]}, {row[keys[0]]})')

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
        centroidColors.append(
            f'rgb({row[keys[2]]}, {row[keys[1]]}, {row[keys[0]]})')

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


def generateSegmentedImage(data, centroids):
    xColumns = list(centroids.columns)
    clusterColumns = list(centroids.index)

    for rowLabel, row in data.iterrows():
        cluster = row[clusterColumns].idxmax()
        data.loc[rowLabel, xColumns] = centroids.loc[cluster]

    return data


def runAnalysis(photo):
    originalImage = readImage(f"./output/{photo}.jpg")

    data = pd.read_csv(f"./output/data_{photo}.csv", index_col=0)
    centroids = pd.read_csv(f"./output/centroids_{photo}.csv", index_col=0)

    xColumns = list(centroids.columns)

    # plotDataAndCentroids3d(data, centroids, ["x0", "x1", "x2"])

    processedData = generateSegmentedImage(data, centroids)
    newImage = np.array(processedData.loc[:, xColumns]).reshape(
        originalImage.shape)
    writeImage(f"./output/segmented_{photo}.jpg", newImage)


# %% run main script
photos = [
    "photo001",
    "photo002",
    "photo003",
    "photo004",
    "photo005",
    "photo006",
    "photo007",
    "photo008",
    "photo009",
    "photo010",
    "photo011",
]

for photo in photos:
    runAnalysis(photo)
