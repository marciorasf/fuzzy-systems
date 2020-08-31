#  %% imports
from scipy.io import loadmat
import pandas as pd
import numpy as np
from c_means import euclNorm
from functools import reduce
import plotly.express as px

# %% initialize input data
nPoints = 100
xDimension = 2
rawData = np.random.normal(0.5, 0.4, (nPoints, xDimension))

xColumns = [f'x{i}' for i in range(xDimension)]
df = pd.DataFrame(rawData, columns=xColumns)

limitsColumns = ["min", "max"]
limits = pd.DataFrame([], columns=limitsColumns)
for col in xColumns:
    limits.loc[col, :] = [df[col].min(), df[col].max()]

fig = px.scatter(df, x="x0", y="x1")
fig.show()

# %% initialize algorithm data

centroids = pd.DataFrame(columns=xColumns)
densities = pd.DataFrame(
    np.zeros((nPoints, 1)),
    columns=["density"]
)


# %% step 1
radiusA = 1
denominatorA = (radiusA/2)**2

for row_label, row in df.iterrows():
    density = reduce(
        lambda sum, x: sum + np.exp(- ((euclNorm(row - x)**2)/denominatorA)),
        df.loc[:, :].to_numpy(),
        0
    )
    densities.loc[row_label, "density"] = density

maxDensityIndex = densities["density"].idxmax()
maxDensity =  densities.loc[maxDensityIndex, "density"]
currentSelectedCentroid = df.iloc[maxDensityIndex]
centroids = centroids.append(currentSelectedCentroid)

# %% step 2
radiusB = 1.5*radiusA
denominatorB = (radiusB/2)**2

for row_label, row in df.iterrows():
    density = np.exp(- ((euclNorm(row - currentSelectedCentroid)**2)/denominatorB))
    densities.loc[row_label, "density"] -= maxDensity*density


maxDensityIndex = densities["density"].idxmax()
maxDensity =  densities.iloc[maxDensityIndex]
currentSelectedCentroid = df.iloc[maxDensityIndex]
centroids = centroids.append(currentSelectedCentroid)

fig = px.scatter(centroids, x="x0", y="x1")
fig.show()
# %%
