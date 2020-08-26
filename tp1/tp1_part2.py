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

    # Add traces
    fig.add_trace(go.Scatter3d(
        x=data[keys[0]], 
        y=data[keys[1]],
        z=data[keys[2]], 
        mode="markers", 
        name="Input points",
        marker=dict(
            color="blue",
            size=2
        )
        ))
    fig.add_trace(go.Scatter3d(
        x=centroids[keys[0]], 
        y=centroids[keys[1]], 
        z=centroids[keys[2]], 
        mode="markers", 
        name="Centroids",
        marker=dict(
            color="red",
            size=15
        )
        ))
    fig.show()


# %%
imageMatrix = imageio.imread("./tp1/data/photo003.jpg")
flatImage = np.array([item for sublist in imageMatrix for item in sublist])
# flatImage = flatImage[0 : : 100]

data, centroids, *_ = cMeans(flatImage, 3, 1e-9, 100, 2)

printDataAndCentroids3d(data,centroids,["x0", "x1", "x2"])
data.to_csv("data.csv", compression=None)
data.to_csv("centroids.csv", compression=None)