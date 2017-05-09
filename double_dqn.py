import tensorflow as tf
import numpy as np
from collections import deque
import random


INITIAL_EPSILON = 0.5
FINAL_EPSILON = 0.01
MID_DIM = 20
REPLAY_SIZE = 10000
BATCH_SIZE = 32
GAMMA = 0.99
P_SCOPE = 'primary'
T_SCOPE = 'target'
ALPHA = 0.0


class QNet(object):
    def __init__(self, i_dim, h_dim, o_dim):
        self.i_dim = i_dim
        self.h_dim = h_dim
        self.o_dim = o_dim
        # w1 = tf.get_variable('w1', shape=[i_dim, h_dim], initializer=tf.random_normal_initializer())
        # b1 = tf.get_variable('b1', shape=[h_dim], initializer=tf.constant_initializer(0.01))
        # w2 = tf.get_variable('w2', shape=[h_dim, o_dim], initializer=tf.random_normal_initializer())
        # b2 = tf.get_variable('b2', shape=[o_dim], initializer=tf.constant_initializer(0.01))
        w1 = tf.Variable(tf.truncated_normal([i_dim, h_dim]))
        b1 = tf.Variable(tf.constant(0.01, shape=[h_dim]))
        w2 = tf.Variable(tf.truncated_normal([h_dim, o_dim]))
        b2 = tf.Variable(tf.constant(0.01, shape=[o_dim]))
        self.state_input = tf.placeholder(tf.float32, shape=[None, i_dim])
        h_layer = tf.nn.relu(tf.matmul(self.state_input, w1) + b1)
        self.q_val = tf.matmul(h_layer, w2) + b2
        self.weights = [w1, b1, w2, b2]

        self.prepare_train()

    def prepare_train(self):
        self.y_input = tf.placeholder(tf.float32, [None])
        self.action_input = tf.placeholder(tf.float32, [None, self.o_dim])
        Q_action = tf.reduce_sum(tf.multiply(self.q_val, self.action_input), reduction_indices=1)
        self.optimizer = tf.train.AdamOptimizer(0.0001).minimize(tf.reduce_mean(tf.square(self.y_input - Q_action)))


class DoubleDqn(object):
    def __init__(self, env):
        self.replay_buffer = deque()

        self.epsilon = INITIAL_EPSILON
        self.state_dim, self.action_dim = env.observation_space.shape[0], env.action_space.n
        self.primary_net = QNet(self.state_dim, MID_DIM, self.action_dim)
        self.target_net = QNet(self.state_dim, MID_DIM, self.action_dim)

        self.update_target_op = [tf.assign(b, ALPHA*b+(1-ALPHA)*a)
                                 for a, b in zip(self.primary_net.weights, self.target_net.weights)]

        self.session = tf.InteractiveSession()
        self.session.run(tf.global_variables_initializer())

        self.episode_num = 0

    def perceive(self, state, action, reward, next_state, done):
        # assert isinstance(action, int)
        one_hot_action = np.zeros(self.action_dim)
        one_hot_action[action] = 1
        self.replay_buffer.append((state, one_hot_action, reward, next_state, done))
        if done:
            self.episode_num += 1
            if not self.episode_num % 10:
                self.session.run(self.update_target_op)
        if len(self.replay_buffer) > REPLAY_SIZE:
            self.replay_buffer.popleft()
        if len(self.replay_buffer) > BATCH_SIZE:
            for i in range(5):
                self.train_Q_network()

    def train_Q_network(self):
        minibatch = random.sample(self.replay_buffer, BATCH_SIZE)
        state_batch = [data[0] for data in minibatch]
        action_batch = [data[1] for data in minibatch]
        reward_batch = [data[2] for data in minibatch]
        next_state_batch = [data[3] for data in minibatch]

        y_batch = []
        Qt_value_batch = self.target_net.q_val.eval(feed_dict={self.target_net.state_input: next_state_batch})
        Qp_value_batch = self.primary_net.q_val.eval(feed_dict={self.primary_net.state_input: next_state_batch})
        for i in range(0, BATCH_SIZE):
            done = minibatch[i][4]
            if done:
                y_batch.append(reward_batch[i])
            else:
                y_batch.append(reward_batch[i] + GAMMA * Qt_value_batch[i][np.argmax(Qp_value_batch[i])])

        self.primary_net.optimizer.run(feed_dict={
            self.primary_net.y_input: y_batch,
            self.primary_net.action_input: action_batch,
            self.primary_net.state_input: state_batch
        })

    def egreedy_action(self, state):
        primary_q = self.session.run(self.primary_net.q_val, feed_dict={self.primary_net.state_input: [state]})[0]

        if random.random() < self.get_epsilon():
            return random.randint(0, self.action_dim-1)
        else:
            return np.argmax(primary_q)

    def action(self, state):
        return np.argmax(self.primary_net.q_val.eval(feed_dict={self.primary_net.state_input: [state]})[0])

    def get_epsilon(self):
        return 1.0/(self.episode_num/200.0 + 2)

    def train(self):
        pass


