import gym
import gym.spaces
from collections import defaultdict
from functools import partial
import numpy as np
import random

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


# Random policy
# action = env.action_space.sample()


####### Make environment #######
env = gym.make('BipedalWalker-v2')

####### Poke around environment (action and observation space) #######
# Four-dimensional action space between -1 and 1
# [pink hip, pink knee, purple hip, purple knee]
# print(env.action_space.high)
# print(env.action_space.low)

# 24-dimensional action space between -inf and inf
# print(env.observation_space)
# print(env.observation_space.high)
# print(env.observation_space.low)

# Discretize action space
action_space_low = env.action_space.low[0]
action_space_high = env.action_space.high[0]
action_space_num = 10
space = np.linspace(action_space_low, action_space_high, action_space_num, endpoint=True)
action_space = [ (a,b,c,d) for a in space for b in space for c in space for d in space ]

# Discretize state space (bounds arbitrarily chosen by me)
state_space_low = -1
state_space_high = 9
def continuous2discrete_state(state):
	discrete_state = []
	for elem in state:
		discrete_state.append( np.clip(np.around(elem, 0), state_space_low, state_space_high) )
	return tuple(discrete_state)

Q = defaultdict(float)

# Find the next action that maximizes Q given the current state
# Return (maximizing action, max Q)
def getBestAction_Q(state):
	best_action = None
	best_Q = -100

	for action in action_space:
		if Q[(state, action)] >= best_Q:
			best_Q = Q[(state, action)]
			best_action = action

	return (best_action, best_Q)

####### Execute iterations #######
rewards = []
timesteps = []
num_episodes = 50
times_fallen = 0
learning_rate = 10
discount_rate = 1
exploration_prob = .5


for episode in range(num_episodes):
	state = continuous2discrete_state(env.reset())
	done = False
	num_timesteps = 0
	total_reward = 0

	while not done and num_timesteps <= 200:
		# Render the last episode
		if (episode == num_episodes - 1):
			env.render()

		action = None
		if random.random() > exploration_prob:
			action, _ = getBestAction_Q(state)
			# print('action:', action)
		else:
			action = random.choice(action_space)

		result_state, reward, done, info = env.step(action)
		result_state = continuous2discrete_state(result_state)
		_, result_state_value = getBestAction_Q(result_state)
		Q[(state, action)] += learning_rate * (reward + discount_rate * result_state_value)
		# print(Q[(state, action)])

		state = result_state
		total_reward += reward
		num_timesteps += 1
		# print('state', state)
		# print('action', action)
		# print('reward', reward)
		# print('info', info)

	print(num_timesteps, ',', total_reward)
	times_fallen = times_fallen + 1 if done else times_fallen
	rewards.append(total_reward)
	timesteps.append(num_timesteps)

print('fell ', float(times_fallen) / num_episodes * 100,'%')

# plt.figure(1)

# plt.subplot(211)
# plt.plot(rewards)
# plt.title('Q learning')
# plt.ylabel('reward')

# plt.subplot(212)
# plt.plot(timesteps)
# plt.ylabel('# timesteps')
# plt.show()