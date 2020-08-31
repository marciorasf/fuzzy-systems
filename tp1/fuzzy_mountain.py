#  %% imports
from scipy.io import loadmat
import pandas as pd
import numpy as np
from c_means import euclNorm
from functools import reduce

# %% initialize data
nPoints = 10
xDimension = 2
rawData = np.random.normal(0.5, 0.4, (nPoints, xDimension))

xColumns = [f'x{i}' for i in range(xDimension)]
df = pd.DataFrame(rawData, columns=xColumns)

limitsColumns = ["min", "max"]
limits = pd.DataFrame([], columns=limitsColumns)
for col in xColumns:
    limits.loc[col, :] = [df[col].min(), df[col].max()]

# %% calculate densities
ra = 1
denominator = (ra/2)**2

centroids = pd.DataFrame(columns=xColumns)
densities = pd.DataFrame(
    np.zeros((nPoints, 1)),
    columns=["density"]
)
test = []
for row_label, row in df.iterrows():
    density = reduce(
        lambda sum, x: sum + np.exp(- ((euclNorm(row - x)**2)/denominator)),
        df.loc[:, :].to_numpy(),
        0
    )
    densities.iloc[row_label, 0] = density

maxDensity = densities["density"].idxmax()
centroids = centroids.append(df.iloc[maxDensity])
