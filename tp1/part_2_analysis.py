# %%
import numpy as np
import pandas as pd
import cv2

# %%
photo = "photo002"

data =  pd.read_csv(f"./output/data_{photo}.csv", index_col=0)
centroids = pd.read_csv(f"./output/centroids_{photo}.csv", index_col=0)

xColumns = list(centroids.columns)
clusterColumns = list(centroids.index)

originalImage = cv2.imread(f"./output/{photo}.jpg")

# %%
for rowLabel, row in data.iterrows():
    cluster = row[clusterColumns].idxmax()
    data.loc[rowLabel, xColumns] = centroids.loc[cluster]

newImage = np.array(data.loc[:, xColumns]).reshape(originalImage.shape)
cv2.imwrite(f"./output/segmented_{photo}.jpg", newImage)