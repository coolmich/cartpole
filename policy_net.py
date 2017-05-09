import numpy as np
import tensorflow as tf
import random

MID_DIM = 20
GAMMA = 0.99
TRAIN_INTERVAL = 10

class PolicyNet(object):
    def __init__(self, env):

        # initialization
        self._s = tf.InteractiveSession()
        self.state_dim, self.action_dim = env.observation_space.shape[0], env.action_space.n
        self.episode_num = 0

        # build the graph
        self._input = tf.placeholder(tf.float32,
                shape=[None, self.state_dim])

        self.hidden1 = tf.contrib.layers.fully_connected(
                inputs=self._input,
                num_outputs=MID_DIM,
                activation_fn=tf.nn.relu,
                weights_initializer=tf.random_normal_initializer)

        self.logits = tf.contrib.layers.fully_connected(
                inputs=self.hidden1,
                num_outputs=self.action_dim,
                activation_fn=None)

        logits = self.logits

        # op to sample an action
        self._sample = tf.reshape(tf.multinomial(logits, 1), [])

        # get log probabilities, add 1e-30 to avoid -inf
        log_prob = tf.log(tf.add(tf.nn.softmax(logits), 1e-30))

        # training part of graph
        self._acts = tf.placeholder(tf.int32)
        self._advantages = tf.placeholder(tf.float32)

        # get log probs of actions from episode
        self.indices = tf.range(0, tf.shape(log_prob)[0]) * tf.shape(log_prob)[1] + self._acts
        self.act_prob = tf.gather(tf.reshape(log_prob, [-1]), self.indices)

        # surrogate loss
        loss = -tf.reduce_sum(tf.multiply(self.act_prob, self._advantages))

        # update
        optimizer = tf.train.RMSPropOptimizer(0.1)
        self._train = optimizer.minimize(loss)

        # self.episode_num = 0
        # self.episode_bank = []
        self.cum_rewards = []
        self.cum_actions = []
        self.cum_states = []
        self.tmp_rewards = []

        self._s.run(tf.global_variables_initializer())

    def perceive(self, state,action,reward,next_state,done):
        self.cum_states.append(state)
        self.cum_actions.append(action)
        self.tmp_rewards.append(reward)
        if done:
            self.cum_rewards.extend([len(self.tmp_rewards)] * len(self.tmp_rewards))
            self.tmp_rewards = []
            self.episode_num += 1

    def train(self):
        advantage_val = (self.cum_rewards - np.mean(self.cum_rewards)) / (np.std(self.cum_rewards) + 1e-10)
        self._train.run(feed_dict={
            self._input: np.array(self.cum_states),
            self._acts: np.array(self.cum_actions),
            self._advantages: np.array(advantage_val)
        })
        self.cum_rewards = []
        self.cum_actions = []
        self.cum_states = []

    def egreedy_action(self, state):
        if random.random() < self.get_epsilon():
            return random.randint(0, self.action_dim-1)
        else:
            return self.action(state)

    def action(self, state):
        val = self._s.run(self._sample, feed_dict={self._input: [state]})
        return val

    def get_epsilon(self):
        return 1.0 / (self.episode_num / 100.0 + 1)
