import plotly
import plotly.plotly as py
import plotly.graph_objs as go
import pandas as pd
import numpy as np

random = pd.read_csv('results/RandomPolicy_1528566594.csv')
# random['rolling_avg_reward'] = random['total_reward'].rolling(300).mean()
naive_q = pd.read_csv('results/QLearning_NaiveDiscretize_1528545432.csv')
# naive_q['rolling_avg_reward'] = naive_q['total_reward'].rolling(300).mean()
smart_q = pd.read_csv('results/QLearning_SmartDiscretize_1528549733.csv')
# smart_q['rolling_avg_reward'] = smart_q['total_reward'].rolling(300).mean()
ac = pd.read_csv('results/ActorCritic_1528507727_corrected8linearepsilon.csv')
# ac['rolling_avg_reward'] = ac['total_reward'].rolling(300).mean()
# acer = pd.read_csv('results/ActorCritic_ExperienceReplay_1528169529.csv')
# acer['rolling_avg_reward'] = acer['total_reward'].rolling(300).mean()
ac2 = pd.read_csv('results/ActorCritic_1528589200.csv')

x = range(0, 5000)

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
		    fillcolor='rgba({}, 0.2)'.format(color),
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
		    fillcolor='rgba({}, 0.2)'.format(color),
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

data = [
	# go.Scatter(
 #        x=x, # assign x as the dataframe column 'x'
 #        y=ac2['total_reward'],
 #        name='Actor critic (correct linear epsilon)',
 #    ),
    # go.Scatter(
    #     x=x, # assign x as the dataframe column 'x'
    #     y=random['rolling_avg_reward'],
    #     name='Random policy'
    # ),
    # go.Scatter(
    #     x=x, # assign x as the dataframe column 'x'
    #     y=naive_q['rolling_avg_reward'],
    #     name='Naive Q learning'
    # ),
    # go.Scatter(
    #     x=x, # assign x as the dataframe column 'x'
    #     y=smart_q['rolling_avg_reward'],
    #     name='Improved Q learning'
    # ),
    # go.Scatter(
    #     x=x, # assign x as the dataframe column 'x'
    #     y=ac['rolling_avg_reward'],
    #     name='Actor critic (old wrong linear epsilon)'
    # ),
    # go.Scatter(
    #     x=x, # assign x as the dataframe column 'x'
    #     y=acer['rolling_avg_reward'],
    #     name='Actor critic, experience replay'
    # ),
]

generatePlots(x, random, 'Random', '25, 25, 25', data, True)
# generatePlots(x, naive_q, 'Naive Q learning', '228, 93, 6', data, True)
# generatePlots(x, smart_q, 'Improved Q learning', '252, 174, 97', data, True)
# generatePlots(x, ac, 'Actor Critic linear', '178, 169, 210', data, True)
generatePlots(x, ac2, 'Actor Critic cubic', '93, 52, 153', data, True)

layout = go.Layout(
    title='Moving Average of Rewards',
    yaxis=dict(title='Episode Reward'),
    xaxis=dict(title='Episodes')
)
fig = go.Figure(data=data, layout=layout)
plotly.offline.plot(fig, filename='reward.html')