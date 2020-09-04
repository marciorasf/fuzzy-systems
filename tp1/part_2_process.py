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
        data, centroids, iterations = cMeans(flatImage, nClusters, 1e-15, 50, 0.4, 2)

        data.to_csv(f"output/data_{photo}.csv", compression=None)
        centroids.to_csv(f"output/centroids_{photo}.csv", compression=None)

    except:
        pass


# %% run main script
nClustersPerPhoto = {
    "photo011": 10,
    # "photo001": 13,
    # "photo002": 6,
    # "photo003": 10,
    # "photo005": 12,
    # "photo006": 5,
    # "photo007": 7,
    # "photo008": 9,
    # "photo009": 12,
    # "photo010": 11,
    # "photo004": 10,
}
scaleRatio = 0.3

for photo, nClusters in nClustersPerPhoto.items():
    runCMeansForPhoto(photo, scaleRatio, nClusters)
