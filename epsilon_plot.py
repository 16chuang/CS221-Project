import plotly
import plotly.plotly as py
import plotly.graph_objs as go
import numpy as np

x = np.linspace(0, 1, 10)

linear = -0.8 * x + 1
cubic = -0.8 * (x - 1) ** 3 + 0.2
quadratic = -1 * (x ** 2) + 1
quadratic_less = -0.5 * (x ** 2) + 1

data = [
	go.Scatter(
	    x=x,
	    y=linear,
	    name='linear',
	    mode = 'lines',
    ),
    go.Scatter(
	    x=x,
	    y=cubic,
	    name='cubic',
	    mode = 'lines',
    ),
    go.Scatter(
	    x=x,
	    y=quadratic,
	    name='quadratic',
	    mode = 'lines',
    ),
    go.Scatter(
	    x=x,
	    y=quadratic_less,
	    name='less quadratic',
	    mode = 'lines',
    )
]

layout = go.Layout(
    title='Epsilon Functions',
    yaxis=dict(title='Epsilon'),
    xaxis=dict(title='p')
)
fig = go.Figure(data=data, layout=layout)
plotly.offline.plot(fig, filename='epsilon.html')