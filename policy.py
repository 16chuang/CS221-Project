import numpy as np
from collections import defaultdict
import random

#############################################
#		 	  	 RL ALGORITHM			 	#
#											#
#				Abstract class 				#
#############################################
class RLAlgorithm:
	def __init__(self, env): raise NotImplementedError("Override me")

	# Produce action given state
	def get_action(self, state): raise NotImplementedError("Override me")

	# Incorporate environment feedback after an action
	def update(self, state, action, reward, new_state): raise NotImplementedError("Override me")

	def get_name(self): raise NotImplementedError("Override me")


#############################################
#		 	  	 RANDOM POLICY			 	#
#											#
# 	  Randomly samples from action space 	#
#############################################
class RandomPolicy(RLAlgorithm):
	def __init__(self, env):
		self.action_space = env.action_space

	def get_action(self, state):
		return self.action_space.sample()

	def update(self, state, action, reward, new_state):
		pass

	def get_name(self):
		return 'RandomPolicy'


#############################################
#		 	    NAIVE Q LEARNING		 	#
#											#
#  Basic Q learning on naively discretized 	#
#  			action and state spaces 	 	#
#############################################
class QLearning_NaiveDiscretize(RLAlgorithm):
	def __init__(self, env):
		# Discretize action space
		action_space_low = env.action_space.low[0]   # -1
		action_space_high = env.action_space.high[0] # +1
		action_space_num = 5
		space = np.linspace(action_space_low, action_space_high, action_space_num, endpoint=True)
		self.action_space = [ (a,b,c,d) for a in space for b in space for c in space for d in space ]

		# Discretized state space bounds
		self.state_space_low = -1
		self.state_space_high = 9

		self.Q = defaultdict(float)

		# Tunable constants
		self.exploration_prob = .2
		self.discount_rate = 1
		self.learning_rate = 10

	# Discretize state space (bounds arbitrarily chosen by me)
	def continuous2discrete_state(self, state):
		discrete_state = []
		for elem in state:
			discrete_state.append( np.clip(np.around(elem, 0), self.state_space_low, self.state_space_high) )
		return tuple(discrete_state)

	# Find the next action that maximizes Q given the current state
	# Return (maximizing action, max Q)
	def getBestAction_Q(self, state):
		best_action = None
		best_Q = -100

		for action in self.action_space:
			if self.Q[(state, action)] >= best_Q:
				best_Q = self.Q[(state, action)]
				best_action = action

		return (best_action, best_Q)

	def get_action(self, state):
		# Sometimes pick random action to explore
		if np.random.random() > self.exploration_prob:
			return random.choice(self.action_space)
		else: # Otherwise find action that maximizes Q
			state = self.continuous2discrete_state(state)
			return self.getBestAction_Q(state)[0]

	def update(self, state, action, reward, new_state):
		state = self.continuous2discrete_state(state)
		new_state = self.continuous2discrete_state(new_state)
		_, new_state_value = self.getBestAction_Q(new_state)
		self.Q[(state, action)] += self.learning_rate * (reward + self.discount_rate * new_state_value)

	def get_name(self):
		return 'QLearning_NaiveDiscretize'




