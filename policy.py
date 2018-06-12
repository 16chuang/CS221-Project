import numpy as np
from collections import defaultdict
import random
from actor_critic import Actor, Critic
import tensorflow as tf
from sklearn.linear_model import SGDRegressor
import sklearn.preprocessing
from sklearn.preprocessing import normalize
import warnings

#############################################
#		 	  	 RL ALGORITHM			 	#
#											#
#				Abstract class 				#
#############################################
class RLAlgorithm:
	def __init__(self, env): raise NotImplementedError("Override me")

	# Produce action given state
	def get_action(self, state, episode_percentage): raise NotImplementedError("Override me")

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

	def get_action(self, state, episode_percentage):
		return self.action_space.sample()

	def update(self, state, action, reward, new_state):
		pass

	def get_name(self):
		return 'RandomPolicy'


#############################################
#		 	   NAIVE Q LEARNING			 	#
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

	def get_action(self, state, episode_percentage):
		# Sometimes pick random action to explore
		if np.random.random() < self.exploration_prob:
			return random.choice(self.action_space)
		else: # Otherwise find action that maximizes Q
			state = self.continuous2discrete_state(state)
			return self.getBestAction_Q(state)[0]

	def update(self, state, action, reward, new_state):
		state = self.continuous2discrete_state(state)
		new_state = self.continuous2discrete_state(new_state)
		_, new_state_value = self.getBestAction_Q(new_state)
		self.Q[(state, action)] = self.learning_rate * (reward + self.discount_rate * new_state_value)

	def get_name(self):
		return 'QLearning_NaiveDiscretize'


#############################################
#		      SMARTER? Q LEARNING		 	#
#											#
#############################################
class QLearning_SmartDiscretize(RLAlgorithm):
	def __init__(self, env):
		# Discretize action space
		# action_space_low = env.action_space.low[0]   # -1
		# action_space_high = env.action_space.high[0] # +1
		action_space_low = -0.4
		action_space_high = 0.4
		action_space_num = 10
		space = np.linspace(action_space_low, action_space_high, action_space_num, endpoint=True)
		self.space = space
		self.action_space = [ (a,b,c,d) for a in space for b in space for c in space for d in space ]

		self.Q = defaultdict(float)

		# Tunable constants
		self.exploration_prob = .2
		self.discount_rate = 1
		self.learning_rate = 1

	# Discretize state space 
	# 	Skip lidar readings
	def continuous2discrete_state(self, state):
		discrete_state = []

		num_buckets = 10

		# Hull angle, hip joint angles (x2), knee joint angles (x2) (0, 2*pi)
		lo = -np.pi
		hi = np.pi
		bucket_size = float(hi-lo) / num_buckets
		# print state[4]
		# print (np.clip(state[4], lo, hi) - lo) / bucket_size
		discrete_state.append( int((np.clip(state[0], lo, hi) - lo) / bucket_size) )
		discrete_state.append( int((np.clip(state[4], lo, hi) - lo) / bucket_size) )
		discrete_state.append( int((np.clip(state[6], lo, hi) - lo) / bucket_size) )
		discrete_state.append( int((np.clip(state[9], lo, hi) - lo) / bucket_size) )
		discrete_state.append( int((np.clip(state[11], lo, hi) - lo) / bucket_size) )

		# # Velocity x, y (-1, 1)
		# lo = -1
		# hi = 1
		# bucket_size = float(hi-lo) / num_buckets
		# discrete_state.append( int((state[2] - lo) / bucket_size) )
		# discrete_state.append( int((state[3] - lo) / bucket_size) )

		# # Hip joint speeds (x2), knee joint speeds (x2)
		# lo = -2
		# hi = 2
		# bucket_size = float(hi-lo) / num_buckets
		# discrete_state.append( int((np.clip(state[5], lo, hi) - lo) / bucket_size) )
		# discrete_state.append( int((np.clip(state[7], lo, hi) - lo) / bucket_size) )
		# discrete_state.append( int((np.clip(state[10], lo, hi) - lo) / bucket_size) )
		# discrete_state.append( int((np.clip(state[12], lo, hi) - lo) / bucket_size) )

		# Ground contact flags
		discrete_state.append( int(state[8]) )
		discrete_state.append( int(state[13]) )

		# print discrete_state
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

	def get_exploration_prob(self, episode_percentage):
		return -1 * (episode_percentage - 1) ** 3

	def get_action(self, state, episode_percentage):
		# Sometimes pick random action to explore
		if np.random.random() < self.get_exploration_prob(episode_percentage):
			return (random.choice(self.space), random.choice(self.space), random.choice(self.space), random.choice(self.space))
		else: # Otherwise find action that maximizes Q
			state = self.continuous2discrete_state(state)
			return self.getBestAction_Q(state)[0]

	def update(self, state, action, reward, new_state):
		state = self.continuous2discrete_state(state)
		new_state = self.continuous2discrete_state(new_state)
		_, new_state_value = self.getBestAction_Q(new_state)
		update = self.learning_rate * (self.Q[(state, action)] - (reward + self.discount_rate * new_state_value))
		self.Q[(state, action)] -= update
		# if (np.abs(self.Q[(state, action)]) != np.abs(update)):
		# 	print('{} {}'.format(update, self.Q[(state, action)]))

	def get_name(self):
		return 'QLearning_SmartDiscretize'
		

class QLearning_FunctionApprox(object):
	def __init__(self, env):
		# Discretize action space
		# action_space_low = -1
		# action_space_high = 1
		# action_space_num = 10
		# space = np.linspace(action_space_low, action_space_high, action_space_num, endpoint=True)
		space = [-.5, -.2, 0, .2, .5]
		self.space = space
		self.action_space = [ (a,b,c,d) for a in space for b in space for c in space for d in space ]

		self.weights = np.ones(14 + 4 + 1)

		# Tunable constants
		self.exploration_prob = .2
		self.discount_rate = 1
		self.learning_rate = .001

		# TF session
		self.r = tf.placeholder(tf.float32, None, 'reward')
		self.v_ = tf.placeholder(tf.float32, [1, 1], "v_next")
		self.input = tf.placeholder(tf.float32, [1, 14+4+1], 'input') # state, action, bias
		
		l1 = tf.layers.dense(
			inputs=self.input,
			units=9,
			activation=tf.nn.relu,
			kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
            bias_initializer=tf.constant_initializer(0.1),  # biases
            name='l1'
		)

		self.Q = tf.layers.dense(
			inputs=l1,
			units=1,
			activation=None,
			kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
            bias_initializer=tf.constant_initializer(0.1),  # biases
            name='Q'
		)

		self.loss = tf.square(self.r + self.discount_rate * self.v_ - self.Q)
		self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

		self.sess = tf.Session()
		self.sess.run(tf.global_variables_initializer())

	def get_Q(self, state, action):
		feature = self.get_feature(state, action)
		q = self.sess.run(self.Q, {self.input: feature})
		return q

	def get_exploration_prob(self, episode_percentage):
		# return -1 * (episode_percentage - 1) ** 3
		return -1 * (episode_percentage ** 2) + 1

	def get_feature(self, state, action): 
		feature = np.concatenate((state[0:14], list(action), [1])) # state, action, bias
		return feature[np.newaxis, :]

	def getBestAction_Q(self, state):
		best_action = None
		best_Q = -100

		for action in self.action_space:
			Q_val = self.get_Q(state, action)
			if Q_val >= best_Q:
				best_Q = Q_val
				best_action = action

		return (best_action, best_Q)

	def get_action(self, state, episode_percentage):
		# Sometimes pick random action to explore
		if np.random.random() < self.get_exploration_prob(episode_percentage):
			return (random.choice(self.space), random.choice(self.space), random.choice(self.space), random.choice(self.space))
		else: # Otherwise find action that maximizes Q
			return self.getBestAction_Q(state)[0]

	def update(self, state, action, reward, new_state):
		v_ = self.getBestAction_Q(new_state)[1]
		Q = self.get_Q(state, action)
		feature = self.get_feature(state, action)

		loss, train_op = self.sess.run([self.loss, self.train_op], 
			{self.r: reward, self.v_: v_, self.Q: Q, self.input: feature})

		# print loss

		# update_coeff = self.learning_rate * (q_opt_state_action - (reward + self.discount_rate * value_opt_new_state))
		# # print 'self.weights', self.weights
		# print 'q_opt_state_action', q_opt_state_action
		# # print '(reward + self.discount_rate * value_opt_new_state)', (reward + self.discount_rate * value_opt_new_state)
		# # print 'self.get_feature(state, action)', self.get_feature(state, action)
		# print 'update_coeff', update_coeff
		# self.weights = self.weights - update_coeff * self.get_feature(state, action)

	def get_name(self):
		return 'QLearning_FunctionApprox'
		

class QLearning_FunctionApprox2(object):
	def __init__(self, env):
		# Discretize action space
		space = [-.5, -.2, 0, .2, .5]
		self.space = space
		self.action_space = [ (a,b,c,d) for a in space for b in space for c in space for d in space ]
		self.action_to_index = defaultdict(int)
		for i in range(len(self.action_space)):
			self.action_to_index[ self.action_space[i] ] = i

		# Tunable constants
		self.exploration_prob = .2
		self.discount_rate = 1
		self.learning_rate = .001

		observation_bounds = np.array([
			[0, -2, -1, -1, -2*np.pi, -2, -2*np.pi, -2, 0, -2*np.pi, -2, -2*np.pi, -2, 0, 0], 
			[2*np.pi, 2, 1, 1, 2*np.pi, 2, 2*np.pi, 2, 1, 2*np.pi, 2, 2*np.pi, 2, 1, 1]
		])
		self.scaler = sklearn.preprocessing.StandardScaler()
		self.scaler.fit(observation_bounds)

		self.models = []
		for _ in self.action_space:
			model = SGDRegressor(loss='squared_loss', learning_rate="constant")
			model.partial_fit([self.get_feature(env.reset())], [0])
			self.models.append(model)

		warnings.filterwarnings("ignore", category=DeprecationWarning)

	def get_exploration_prob(self, episode_percentage):
		# return -1 * (episode_percentage - 1) ** 3
		return -1 * (episode_percentage ** 2) + 1

	def get_feature(self, state): 
		feature = np.concatenate((state[0:14], [1])) # state, action, bias
		scaled = self.scaler.transform(feature)
		normalized = normalize(scaled)
		return normalized[0]

	def predict(self, state, action_index=None):
		feature = self.get_feature(state)

		if action_index == None:
			return np.array([m.predict(feature) for m in self.models])
		else:
			return self.models[action_index].predict(feature)

	def get_action(self, state, episode_percentage):
		# Sometimes pick random action to explore
		if np.random.random() < self.get_exploration_prob(episode_percentage):
			return (random.choice(self.space), random.choice(self.space), random.choice(self.space), random.choice(self.space))
		else: # Otherwise find action that maximizes Q
			best_action_index = np.argmax(self.predict(state))
			return self.action_space[best_action_index]

	def update(self, state, action, reward, new_state):
		v_ = np.max(self.predict(new_state))

		action_index = self.action_to_index[action]
		feature = self.get_feature(state)
		td_error = reward + self.discount_rate * v_
		self.models[action_index].partial_fit([feature], [td_error])

		# update_coeff = self.learning_rate * (q_opt_state_action - (reward + self.discount_rate * value_opt_new_state))
		# # print 'self.weights', self.weights
		# print 'q_opt_state_action', q_opt_state_action
		# # print '(reward + self.discount_rate * value_opt_new_state)', (reward + self.discount_rate * value_opt_new_state)
		# # print 'self.get_feature(state, action)', self.get_feature(state, action)
		# print 'update_coeff', update_coeff
		# self.weights = self.weights - update_coeff * self.get_feature(state, action)

	def get_name(self):
		return 'QLearning_FunctionApprox'


#############################################
#		  	    ACTOR CRITIC			 	#
#											#
#############################################
class ActorCritic(object):
	def __init__(self, env):
		LR_A = 0.001    # learning rate for actor
		LR_C = 0.01     # learning rate for critic
		num_features = env.observation_space.shape[0]
		# num_features = 14
		num_actions = env.action_space.shape[0]

		self.action_space = env.action_space

		sess = tf.Session()
		self.actor = Actor(sess, n_features=num_features, action_bound=[env.action_space.low[0], env.action_space.high[0]], lr=LR_A)
		self.critic = Critic(sess, n_features=num_features, lr=LR_C)     # we need a good teacher, so the teacher should learn faster than the actor
		sess.run(tf.global_variables_initializer())

	def get_action(self, state, episode_percentage):
		# state = state[0:14]

		# Sometimes pick random action to explore
		if np.random.random() < self.get_exploration_prob(episode_percentage):
			# print 'random'
			return self.action_space.sample()
		else:
			# print 'not random'
			return self.actor.choose_action(state)[0]

	def get_exploration_prob(self, episode_percentage):
		# if (episode_percentage > .8):
		# 	epsilon = 0.3
		# else:
		epsilon = -1 * (episode_percentage ** 2) + 1
		# epsilon = -1 * (episode_percentage - 1) ** 3
		# epsilon = -0.8 * (episode_percentage - 1) ** 3 + 0.2
		# epsilon = -0.8 * episode_percentage + 1
		# print epsilon
		return epsilon

	def update(self, state, action, reward, new_state):
		# state = state[0:14]
		# new_state = new_state[0:14]

		td_error = self.critic.learn(state, reward, new_state)  # gradient = grad[r + gamma * V(s_) - V(s)]
		# print td_error
		self.actor.learn(state, action, td_error)     # true_gradient = grad[logPi(s,a) * td_error]

	def get_name(self):
		return 'ActorCritic'



#############################################
#	   ACTOR CRITIC + EXPERIENCE REPLAY 	#
#											#
#############################################
class ActorCriticExperienceReplay(object):
	def __init__(self, env):
		self.MEMORY_SIZE = 200
		self.BATCH_SIZE = 10

		LR_A = 0.001    # learning rate for actor
		LR_C = 0.01     # learning rate for critic
		num_features = env.observation_space.shape[0]
		num_actions = env.action_space.shape[0]

		self.action_space = env.action_space

		sess = tf.Session()
		self.actor = Actor(sess, n_features=num_features, action_bound=[env.action_space.low[0], env.action_space.high[0]], lr=LR_A)
		self.critic = Critic(sess, n_features=num_features, lr=LR_C)     # we need a good teacher, so the teacher should learn faster than the actor
		sess.run(tf.global_variables_initializer())

		self.replay_memory = []

	def get_action(self, state, episode_percentage):
		# Sometimes pick random action to explore
		if np.random.random() < self.get_exploration_prob(episode_percentage):
			return self.action_space.sample()
		else:
			return self.actor.choose_action(state)[0]

	def get_exploration_prob(self, episode_percentage):
		return -1 * (episode_percentage ** 2) + 1
		# return -1 * (episode_percentage - 1) ** 3

	def update(self, state, action, reward, new_state):
		td_error = self.critic.learn(state, reward, new_state)  # gradient = grad[r + gamma * V(s_) - V(s)]
		self.actor.learn(state, action, td_error)     # true_gradient = grad[logPi(s,a) * td_error]

		# Add to replay memory
		self.replay_memory.append((state, action, reward, new_state))
		if len(self.replay_memory) >= self.MEMORY_SIZE:
			self.replay_memory.pop(0)

		# Learn from replayed memories
		if np.random.random() < 0.5 and len(self.replay_memory) > self.BATCH_SIZE:
			minibatch = random.sample(self.replay_memory, self.BATCH_SIZE)
			for (batch_state, batch_action, batch_reward, batch_new_state) in minibatch:
				td_error = self.critic.learn(batch_state, batch_reward, batch_new_state)
				self.actor.learn(batch_state, batch_action, td_error)

	def get_name(self):
		return 'ActorCritic_ExperienceReplay'
