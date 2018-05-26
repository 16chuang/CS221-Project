import gym
import gym.spaces
from functools import partial
import time
import numpy as np
import random

from policy import RandomPolicy, QLearning_NaiveDiscretize

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

# # Discretize action space
# action_space_low = env.action_space.low[0]
# action_space_high = env.action_space.high[0]
# action_space_num = 5
# space = np.linspace(action_space_low, action_space_high, action_space_num, endpoint=True)
# action_space = [ (a,b,c,d) for a in space for b in space for c in space for d in space ]



# policy = RandomPolicy(env)
policy = QLearning_NaiveDiscretize(env)

output_file = open("results/{}_{}.csv".format(policy.get_name(), int(time.time())), "w+")
output_file.write(('{},{},{}\n').format('num_timesteps', 'total_reward', 'fell'))

#############################################
#		 	   Execute episodes 	 	    #
#############################################
rewards = []
timesteps = []
num_episodes = 300
times_fallen = 0

for episode in range(num_episodes):
	state = env.reset()
	done = False
	num_timesteps = 0
	total_reward = 0

	if episode % (num_episodes / 5) == 0:
		print('============ EPISODE {} ============').format(episode)

	while not done and num_timesteps <= 200:
		# Render the last episode
		if (episode == num_episodes - 1):
			env.render()

		# Select action
		action = policy.get_action(state)

		# Perform action
		result_state, reward, done, info = env.step(action)
		policy.update(state, action, reward, result_state)

		state = result_state
		total_reward += reward
		num_timesteps += 1
		# print('state', state)
		# print('action', action)
		# print('reward', reward)
		# print('info', info)

	print('num timesteps: {}, reward: {}').format(num_timesteps, total_reward)
	times_fallen = times_fallen + 1 if done else times_fallen
	rewards.append(total_reward)
	timesteps.append(num_timesteps)
	output_file.write(('{},{},{}\n').format(num_timesteps, total_reward, done))

#############################################
#		 	  	Report results	 	    	#
#############################################
output_file.close()

print('fell {}%').format(float(times_fallen) / num_episodes * 100)

plt.figure(1)

plt.subplot(211)
plt.plot(rewards)
plt.title('Q learning')
plt.ylabel('reward')

plt.subplot(212)
plt.plot(timesteps)
plt.ylabel('# timesteps')
plt.show()