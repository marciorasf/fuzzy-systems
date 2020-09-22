from skfuzzy import gaussmf
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

x = np.arange(1,100)
youngMembership = gaussmf(x, 0, 20)
oldMembership = gaussmf(x, 100, 30)

fig = go.Figure()
fig.add_trace(go.Scatter(x=x, y=youngMembership, name="young"))
fig.add_trace(go.Scatter(x=x, y=oldMembership, name="old"))
fig.show()

