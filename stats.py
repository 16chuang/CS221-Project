import pandas as pd

random = pd.read_csv('results/RandomPolicy_1528566594.csv')
naive_q = pd.read_csv('results/QLearning_NaiveDiscretize_1528545432.csv')
smart_q = pd.read_csv('results/QLearning_SmartDiscretize_1528549733.csv')
ac = pd.read_csv('results/ActorCritic_1528593087_correctedquadraticepsilon.csv')
acer = pd.read_csv('results/ActorCritic_ExperienceReplay_1528662900_quadratic5000.csv')
function_approx = pd.read_csv('results/QLearning_FunctionApprox_1528751087_normalized_squaredloss.csv')

df = function_approx
print 'avg', df['total_reward'].mean()
print 'max', df['total_reward'].max()
print 'last window avg', df['total_reward'][-300:].mean()