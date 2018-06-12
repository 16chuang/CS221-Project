import gym
import gym.spaces
from functools import partial
import time
import numpy as np
import random

from policy import RandomPolicy, QLearning_NaiveDiscretize, \
 QLearning_SmartDiscretize, QLearning_FunctionApprox, QLearning_FunctionApprox2, \
 ActorCritic, ActorCriticExperienceReplay

import matplotlib
# matplotlib.use('TkAgg')
from matplotlib import pyplot as plt

import tensorflow as tf
from actor_critic import Actor, Critic


#############################################
#		 	   Make environment		 	    #
#############################################
env = gym.make('BipedalWalker-v2')
env.seed(1)

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

num_features = env.observation_space.shape[0]
num_actions = env.action_space.shape[0]
print 'num_features', num_features
print 'num_actions', num_actions

WRITE_TO_FILE = True

# policy = RandomPolicy(env)
# policy = QLearning_NaiveDiscretize(env)
# policy = QLearning_SmartDiscretize(env)
# policy = QLearning_FunctionApprox(env)
# policy = ActorCritic(env)
# policy = ActorCriticExperienceReplay(env)

def run(policy):
	filename = "results/{}_{}.csv".format(policy.get_name(), int(time.time()))

	if WRITE_TO_FILE:
		with open(filename, "a+") as output_file:
			output_file.write(('{},{},{}\n').format('num_timesteps', 'total_reward', 'fell'))
		# output_file = open("results/{}_{}.csv".format(policy.get_name(), int(time.time())), "w+")

	#############################################
	#		 	   Execute episodes 	 	    #
	#############################################
	rewards = []
	avg_rewards = []
	timesteps = []
	MAX_EPISODES = 2000
	MAX_EP_STEPS = 1000   # maximum time step in one episode
	times_fallen = 0
	avg_reward = None

	for episode in range(MAX_EPISODES):
		state = env.reset()
		done = False
		num_timesteps = 0
		total_reward = 0

		while not done and num_timesteps <= MAX_EP_STEPS:
			# if (episode > 600):
			# if total_reward > 1000:
				# env.render()

			# Select action
			action = policy.get_action(state, float(episode)/MAX_EPISODES)
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

		# avg_reward = 0.9 * avg_reward + 0.1 * total_reward if avg_reward != None else total_reward
		# print('num timesteps: {}, reward: {}, avg_reward: {}').format(num_timesteps, total_reward, avg_reward)
		print('episode: {}, num timesteps: {}, reward: {}').format(episode, num_timesteps, total_reward)
		times_fallen = times_fallen + 1 if done else times_fallen
		if WRITE_TO_FILE:
			with open(filename, "a+") as output_file:
				output_file.write(('{},{},{}\n').format(num_timesteps, total_reward, done))

	#############################################
	#		 	  	Report results	 	    	#
	#############################################
	if WRITE_TO_FILE:
		output_file.close()

	print('fell {}%').format(float(times_fallen) / MAX_EPISODES * 100)

run(QLearning_FunctionApprox2(env))