import gym
import gym.spaces
from functools import partial
import time
import numpy as np
import random

from policy import RandomPolicy, QLearning_NaiveDiscretize, QLearning_SmartDiscretize

import matplotlib
# matplotlib.use('TkAgg')
from matplotlib import pyplot as plt


#############################################
#		 	   Make environment		 	    #
#############################################
env = gym.make('BipedalWalker-v2')

#############################################
#		 	Poke around environment	 	    #
#		(action and observation space) 	    #
#############################################
# Four-dimensional action space between -1 and 1
# [pink hip, pink knee, purple hip, purple knee]
# print(env.action_space.high)
# print(env.action_space.low)

# 24-dimensional action space between -inf and inf
# print(env.observation_space)
# print(env.observation_space.high)
# print(env.observation_space.low)

WRITE_TO_FILE = False

# policy = RandomPolicy(env)
# policy = QLearning_NaiveDiscretize(env)
policy = QLearning_SmartDiscretize(env)

if WRITE_TO_FILE:
	output_file = open("results/{}_{}.csv".format(policy.get_name(), int(time.time())), "w+")
	output_file.write(('{},{},{}\n').format('num_timesteps', 'total_reward', 'fell'))

#############################################
#		 	   Execute episodes 	 	    #
#############################################
rewards = []
avg_rewards = []
timesteps = []
num_episodes = 500
times_fallen = 0
avg_reward = None

for episode in range(num_episodes):
	state = env.reset()
	done = False
	num_timesteps = 0
	total_reward = 0

	if episode % (num_episodes / 5) == 0:
		print('============ EPISODE {} ============').format(episode)

	while not done: # and num_timesteps <= 500:
		# env.render()
		# Render the last episode
		# if (episode == num_episodes - 1):
		# 	env.render()

		# Select action
		action = policy.get_action(state, float(episode)/num_episodes)
		# print action

		# Perform action
		result_state, reward, done, info = env.step(action)
		if (reward > 0):
			reward *= 50
		# print(reward)
		policy.update(state, action, reward, result_state)

		state = result_state
		total_reward += reward
		num_timesteps += 1
		# print('state', state)
		# print('action', action)
		# print('reward', reward)
		# print('info', info)

	avg_reward = 0.9 * avg_reward + 0.1 * total_reward if avg_reward != None else total_reward
	print('num timesteps: {}, reward: {}, avg_reward: {}').format(num_timesteps, total_reward, avg_reward)
	times_fallen = times_fallen + 1 if done else times_fallen
	avg_rewards.append(avg_reward)
	rewards.append(total_reward)
	timesteps.append(num_timesteps)
	if WRITE_TO_FILE:
		output_file.write(('{},{},{}\n').format(num_timesteps, total_reward, done))

#############################################
#		 	  	Report results	 	    	#
#############################################
if WRITE_TO_FILE:
	output_file.close()

print('fell {}%').format(float(times_fallen) / num_episodes * 100)

plt.figure(1)

plt.subplot(211)
plt.plot(rewards, 'b')
plt.plot(avg_rewards, 'r')
plt.title(('{}').format(policy.get_name()))
plt.ylabel('reward')

plt.subplot(212)
plt.plot(timesteps)
plt.ylabel('# timesteps')
plt.show()