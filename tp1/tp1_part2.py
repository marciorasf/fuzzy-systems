# %%
from scipy.io import loadmat
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from functools import reduce
from c_means import cMeans
import imageio

# %%
def printDataAndCentroids3d(data, centroids, keys):
    fig = make_subplots(x_title="x", y_title="y")

    pointColors = []
    for _, row in data.iterrows():
        pointColors.append(f'rgb({row["x0"]}, {row["x1"]}, {row["x2"]})')

    # Add traces
    fig.add_trace(go.Scatter3d(
        x=data[keys[0]], 
        y=data[keys[1]],
        z=data[keys[2]], 
        mode="markers", 
        name="Input points",
        opacity=0.5,
        marker=dict(
            color=pointColors,
            size=3
        )
        ))

    centroidColors = []
    for _, row in centroids.iterrows():
        centroidColors.append(f'rgb({row["x0"]}, {row["x1"]}, {row["x2"]})')


    fig.add_trace(go.Scatter3d(
        x=centroids[keys[0]], 
        y=centroids[keys[1]], 
        z=centroids[keys[2]], 
        mode="markers", 
        name="Centroids",
        marker=dict(
            color=centroidColors,
            size=15
        )
        ))
    fig.show()


# %%
photo = "photo006"
imageMatrix = imageio.imread(f"./data/{photo}.jpg")
print(imageMatrix.shape)
flatImage = np.array([item for sublist in imageMatrix for item in sublist])
flatImage = flatImage[0 : : 1]

# %%
data, centroids, iterations = cMeans(flatImage, 3, 1e-6, 100, 2)

#  %%
printDataAndCentroids3d(data,centroids,["x0", "x1", "x2"])
print(iterations)
print(centroids)
data.to_csv(f"data_{photo}.csv", compression=None)
centroids.to_csv(f"centroids_{photo}.csv", compression=None)