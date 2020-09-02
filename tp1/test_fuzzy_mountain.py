# %% imports
from fuzzy_mountain import fuzzyMountain, formatData
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from image_utils import readImage, rescaleImage, writeImage
from c_means import cMeans

# %%
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

# %% test 1
# nPoints = 10000
# xDimension = 2
# rawData = np.random.normal(0.5, 0.4, (nPoints, xDimension))

# data = formatData(rawData)

# nClusters = 5
# centroids = fuzzyMountain(data, nClusters)
# print(centroids)


# %% test 2
photo = "photo005"
scaleRatio = 0.05

tempImage = readImage(f"./input/{photo}.jpg")
image = rescaleImage(tempImage, scaleRatio)
writeImage(f"./test/{photo}.jpg", image)

flatImage = np.array([item for sublist in image for item in sublist])
data, centroids, _ = cMeans(flatImage, 12, 1e-5, 1, .2, 2)

# %%
centroidsCasted = centroids.astype(np.int32)
plotDataAndCentroids3d(data, centroidsCasted, ["x0", "x1", "x2"])

# %%
