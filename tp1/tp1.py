# %%
from scipy.io import loadmat
import pandas as pd
import plotly.express as px
import numpy as np

# %%
data = loadmat('./K-Means/SyntheticDataset.mat')
df = pd.DataFrame(data["x"], columns=["x", "y"])
df.labes = ["x, y"]
px.scatter(df, x="x", y="y")

nClusters = 4
for k in range(nClusters):
  df.insert(
    loc=2+k,
    column=f'k{k}',
    value=0
  )
  
# %%
for row_label, row in df.iterrows():
  df.loc[row_label, 2:7] = np.random.dirichlet(np.ones(nClusters), size=1)[0]

# %%
