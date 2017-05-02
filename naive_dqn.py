import tensorflow as tf
from collections import deque
import random
import numpy as np

INITIAL_EPSILON = 0.5
FINAL_EPSILON = 0.01
MID_DIM = 20
REPLAY_SIZE = 10000
BATCH_SIZE = 32
GAMMA = 0.9

class NaiveDqn(object):
    def __init__(self, env):
        self.replay_buffer = deque()

        self.epsilon = INITIAL_EPSILON
        self.state_dim, self.action_dim = env.observation_space.shape[0], env.action_space.n
        self.episode_num = 0
        # self.performance = 0

        self.create_Q_network()
        self.create_training_method()

        self.session = tf.InteractiveSession()
        self.session.run(tf.global_variables_initializer())
        tf.set_random_seed(1234)

    def create_Q_network(self):
        W1 = self._create_weight([self.state_dim, MID_DIM])
        B1 = self._create_bias(shape=[MID_DIM])
        W2 = self._create_weight([MID_DIM, self.action_dim])
        B2 = self._create_bias(shape=[self.action_dim])

        self.state_input = tf.placeholder(tf.float32, shape=[None, self.state_dim])
        h_layer = tf.nn.relu(tf.matmul(self.state_input, W1) + B1)
        self.Q_val = tf.matmul(h_layer, W2) + B2


    def _create_weight(self, shape):
        return tf.Variable(tf.truncated_normal(shape))

    def _create_bias(self, shape):
        return tf.Variable(tf.constant(0.01, shape=shape))

    def create_training_method(self):
        self.y_input = tf.placeholder(tf.float32, [None])
        self.action_input = tf.placeholder(tf.float32, [None, self.action_dim])
        Q_action = tf.reduce_sum(tf.multiply(self.Q_val, self.action_input), reduction_indices=1)
        self.cost = tf.reduce_mean(tf.square(self.y_input - Q_action))
        self.optimizer = tf.train.AdamOptimizer(0.0001).minimize(self.cost)


    def perceive(self, state, action, reward, next_state, done):
        # assert isinstance(action, int)
        one_hot_action = np.zeros(self.action_dim)
        one_hot_action[action] = 1
        self.replay_buffer.append((state, one_hot_action, reward, next_state, done))
        if done:
            self.episode_num += 1
        if len(self.replay_buffer) > REPLAY_SIZE:
            self.replay_buffer.popleft()
        if len(self.replay_buffer) > BATCH_SIZE:
            self.train_Q_network()

    def train_Q_network(self):
        minibatch = random.sample(self.replay_buffer, BATCH_SIZE)
        state_batch = [data[0] for data in minibatch]
        action_batch = [data[1] for data in minibatch]
        reward_batch = [data[2] for data in minibatch]
        next_state_batch = [data[3] for data in minibatch]

        # Q_value_batch = self.Q_val.eval(feed_dict={self.state_input: next_state_batch})
        # y_input = [reward_batch[i] + (0 if minibatch[i][4] else GAMMA * np.max(Q_value_batch[i])) for i in range(BATCH_SIZE)]

        y_batch = []
        Q_value_batch = self.Q_val.eval(feed_dict={self.state_input: next_state_batch})
        for i in range(0, BATCH_SIZE):
            done = minibatch[i][4]
            if done:
                y_batch.append(reward_batch[i])
            else:
                y_batch.append(reward_batch[i] + GAMMA * np.max(Q_value_batch[i]))

        self.optimizer.run(feed_dict={
            self.y_input: y_batch,
            self.action_input: action_batch,
            self.state_input: state_batch
        })

    def egreedy_action(self, state):
        # TODO inspect Q_val
        Q_val = self.Q_val.eval(feed_dict={self.state_input: [state]})[0]
        # self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / 10000
        if random.random() < self.get_epsilon(): #self.epsilon:
            return random.randint(0, self.action_dim-1)
        else:
            return np.argmax(Q_val)

    def action(self, state):
        return np.argmax(self.Q_val.eval(feed_dict = {self.state_input: [state]})[0])

    # def set_performance(self, performance):
    #     self.performance = performance

    def get_epsilon(self):
        return 1.0/(self.episode_num/100.0 + 1)
        # return (200 - self.performance)/200.0