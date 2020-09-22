from skfuzzy.membership import gaussmf
from skfuzzy import fuzzy_and, fuzzy_not
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

universe = np.arange(1, 100)
veryYoungMembership = gaussmf(universe, 0, 20)**2
veryOldMembership = gaussmf(universe, 100, 30)**2

_, notVeryYoungNeitherVeryOldMembership = fuzzy_and(
    universe, fuzzy_not(veryYoungMembership), universe, fuzzy_not(veryOldMembership))

_, veryYoungAndVeryOldMembership = fuzzy_and(
    universe, veryYoungMembership, universe, veryOldMembership)

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=universe,
    y=notVeryYoungNeitherVeryOldMembership,
    name="not very young and not very old"
))

fig.add_trace(go.Scatter(
    x=universe,
    y=veryYoungAndVeryOldMembership,
    name="very young and very old"
))

fig.show()
