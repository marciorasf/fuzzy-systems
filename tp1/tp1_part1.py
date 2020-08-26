# %%
from scipy.io import loadmat
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from functools import reduce
from c_means import cMeans

# %% Declare functions
def printDataAndCentroids(data, centroids, keys):
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
rawData = loadmat("./data/fcm_dataset.mat")

# %% Run C-Means
data, centroids = cMeans(rawData["x"], 4, 10, 2)

printDataAndCentroids(data, centroids, ["x0", "x1"])
