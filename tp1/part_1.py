# %%
from scipy.io import loadmat
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from c_means import cMeans

# %% declare functions
def plotDataAndCentroids(data, centroids, keys):
    fig = make_subplots(x_title="x", y_title="y")

    # add traces
    fig.add_trace(
        go.Scatter(x=data["x0"], y=data["x1"], mode="markers", name="Input points")
    )
    
    fig.add_trace(
        go.Scatter(
            x=centroids["x0"],
            y=centroids["x1"],
            mode="markers",
            name="Centroids",
            marker=dict(color="red", size=15),
        )
    )
    fig.show()


# %% load data from file
rawData = loadmat("./input/fcm_dataset.mat")
rawData = np.array(rawData["x"])

# %% run C-Means
data, centroids, iterations = cMeans(rawData, 4, 1e-12, 50, 0.1, 2)
plotDataAndCentroids(data, centroids, ["x0", "x1"])
