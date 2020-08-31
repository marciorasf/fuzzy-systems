from fuzzy_mountain import fuzzyMountain, formatData
import numpy as np

# %% main script
nPoints = 10
xDimension = 2
rawData = np.random.normal(0.5, 0.4, (nPoints, xDimension))

data = formatData(rawData)

nClusters = 5
centroids = fuzzyMountain(data, nClusters)
print(centroids)