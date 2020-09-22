from skfuzzy import fuzzy_and, fuzzy_not, gaussmf, trapmf, trimf
import numpy as np
import plotly.graph_objects as go

xUniverse = np.arange(1,10, 0.1)
yUniverse = np.arange(1,10, 0.1)

A1Membership = trapmf(xUniverse, [3,4,5,6])
A2Membership = trapmf(xUniverse, [6, 6.5, 7, 7.5])
C1Membership = trimf(yUniverse, [3, 4, 5])
C2Membership = trimf(yUniverse, [4, 5, 6])
ALineMembership = trimf(xUniverse, [5,  6, 7])

fig1 = go.Figure()

fig1.add_trace(go.Scatter(
    x=xUniverse,
    y=A1Membership,
    name="A1"
))
fig1.add_trace(go.Scatter(
    x=yUniverse,
    y=C1Membership,
    name="C1"
))
fig1.add_trace(go.Scatter(
    x=xUniverse,
    y=ALineMembership,
    name="A'"
))


fig1.show()

fig2 = go.Figure()

fig2.add_trace(go.Scatter(
    x=xUniverse,
    y=A2Membership,
    name="A2"
))
fig2.add_trace(go.Scatter(
    x=yUniverse,
    y=C2Membership,
    name="C2"
))
fig2.add_trace(go.Scatter(
    x=xUniverse,
    y=ALineMembership,
    name="A'"
))

fig2.show()

fig3 = go.Figure()

fig3.add_trace(go.Scatter(
    x=xUniverse,
    y=ALineMembership,
    name="A'"
))

fig3.show()
