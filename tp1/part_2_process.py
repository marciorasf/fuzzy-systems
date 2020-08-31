# %% imports
import pandas as pd
import numpy as np
from c_means import cMeans
from image_utils import readImage, rescaleImage, writeImage

# %% declare functions


def runCMeansForPhoto(photo, scaleRatio, nClusters):
    try:
        tempImage = readImage(f"./input/{photo}.jpg")
        image = rescaleImage(tempImage, scaleRatio)
        writeImage(f"./output/{photo}.jpg", image)

        flatImage = np.array([item for sublist in image for item in sublist])
        data, centroids, _ = cMeans(flatImage, nClusters, 1e-5, 50, 0.2, 2)

        data.to_csv(f"output/data_{photo}.csv", compression=None)
        centroids.to_csv(f"output/centroids_{photo}.csv", compression=None)

    except:
        pass


# %% run main script
nClustersPerPhoto = {
    "photo001": 7,
    "photo002": 4,
    "photo003": 7,
    "photo004": 10,
    "photo005": 10,
    "photo006": 4,
    "photo007": 5,
    "photo008": 9,
    "photo009": 12,
    "photo010": 11,
    "photo011": 3,
}
scaleRatio = 0.5

for photo, nClusters in nClustersPerPhoto.items():
    runCMeansForPhoto(photo, scaleRatio, nClusters)
