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
imageMatrix = imageio.imread("./data/photo002.jpg")
flatImage = np.array([item for sublist in imageMatrix for item in sublist])

print(flatImage.shape)

data, centroids = cMeans(flatImage, 2)


# %%
