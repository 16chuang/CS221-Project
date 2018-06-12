import plotly
import plotly.plotly as py
import plotly.graph_objs as go
import pandas as pd
import numpy as np

def generatePlots(x, df, name, color, plots, show_quantiles):
	df['rolling_avg_reward'] = df['total_reward'].rolling(300).mean()

	if show_quantiles:
		df['lower_reward'] = df['total_reward'].rolling(300).quantile(.3)
		df['upper_reward'] = df['total_reward'].rolling(300).quantile(.7)

		trace = go.Scatter(
		    x=x,
		    y=df['rolling_avg_reward'],
		    name=name,
		    mode='lines',
		    line=dict(color='rgb({})'.format(color)),
		    fillcolor='rgba({}, 0.1)'.format(color),
		    fill='tonexty'
		)

		lower_bound = go.Scatter(
		    x=x,
		    y=df['lower_reward'],
		    marker=dict(color="444"),
		    line=dict(width=0),
		    mode='lines',
		    showlegend=False,
	    )

		upper_bound = go.Scatter(
		    x=x,
		    y=df['upper_reward'],
		    mode='lines',
		    marker=dict(color="444"),
		    line=dict(width=0),
		    fillcolor='rgba({}, 0.1)'.format(color),
		    fill='tonexty',
		    showlegend=False
	    )

		plots += [lower_bound, trace, upper_bound]
	else:
		trace = go.Scatter(
	        x=x, # assign x as the dataframe column 'x'
	        y=df['rolling_avg_reward'],
	        name=name
	    )

		plots.append(trace)

data = []

random = pd.read_csv('results/RandomPolicy_1528566594.csv')
naive_q = pd.read_csv('results/QLearning_NaiveDiscretize_1528545432.csv')
smart_q = pd.read_csv('results/QLearning_SmartDiscretize_1528549733.csv')
ac = pd.read_csv('results/ActorCritic_1528593087_correctedquadraticepsilon.csv')
acer = pd.read_csv('results/ActorCritic_ExperienceReplay_1528701965.csv')
function_approx = pd.read_csv('results/QLearning_FunctionApprox_1528751087_normalized_squaredloss.csv')

x = range(0, 2000)

generatePlots(x, random, 'Random', '25, 25, 25', data, True)
generatePlots(x, naive_q, 'Naive Q learning', '228, 93, 6', data, True)
generatePlots(x, smart_q, 'Improved Q learning', '252, 174, 97', data, True)
generatePlots(x, function_approx, 'Q learning with function approximation', '193, 0, 137', data, True)
# generatePlots(x, ac, 'Actor Critic, 40 and 30 unit HL', '93, 52, 153', data, True)
# generatePlots(x, ac2, 'Actor Critic, 80 and 60 unit HL', '178, 169, 210', data, True)
# generatePlots(x, acer, 'Actor Critic Experience Replay', '20, 52, 200', data, True)


# Comparing epsilons
# const = pd.read_csv('results/ActorCritic_1528594776_constepsilon78.csv')
# linear = pd.read_csv('results/ActorCritic_1528507727_corrected8linearepsilon.csv')
# cubic = pd.read_csv('results/ActorCritic_1528509301_correctedcubicepsilon.csv')
# quadratic = pd.read_csv('results/ActorCritic_1528593087_correctedquadraticepsilon.csv')
# generatePlots(x, const, 'Constant (0.8)', '135, 80, 167', data, True)
# generatePlots(x, quadratic, 'Quadratic', '83, 26, 143', data, True)
# generatePlots(x, linear, 'Linear', '102, 137, 207', data, True)
# generatePlots(x, cubic, 'Cubic', '81, 148, 198', data, True)

layout = go.Layout(
    title='Moving Average of Rewards',
    yaxis=dict(title='Episode Reward'),
    xaxis=dict(title='Episodes'),
    legend=dict(orientation="h",x=-0, y=-0.2)
)
fig = go.Figure(data=data, layout=layout)
plotly.offline.plot(fig, filename='reward.html')