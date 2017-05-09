from policy_net import *
import gym

# Hyper Parameters
ENV_NAME = 'CartPole-v0'
EPISODE = 1000000 # Episode limitation
STEP = 300 # Step limitation in an episode
TEST = 10 # The number of experiment test every 100 episode

def policy_rollout(env, agent):
    """Run one episode."""

    observation, reward, done = env.reset(), 0, False
    obs, acts, rews = [], [], []

    while not done:

        # env.render()
        obs.append(observation)

        action = agent.action(observation)
        observation, reward, done, _ = env.step(action)

        acts.append(action)
        rews.append(reward)

    return obs, acts, rews

def main():
  # initialize OpenAI Gym env and dqn agent
  env = gym.make(ENV_NAME)
  agent = PolicyNet(env)

  for episode in xrange(EPISODE):
    # print episode
    # initialize task
    state = env.reset()
    # Train
    for step in xrange(STEP):
      action = agent.egreedy_action(state) # e-greedy action for train
      next_state,reward,done,_ = env.step(action)
      # Define reward for agent
      #reward_agent = reward#-1 if done and step < 150 else 0.1
      agent.perceive(state,action,reward,next_state,done)
      state = next_state
      if done:
        break
    # Test every 100 episodes
    # obs, acts, rews = policy_rollout(env, agent)
    # agent.perceive(obs, acts, rews)
    if episode % 10 == 0:
      agent.train()
      total_reward = 0
      for i in xrange(TEST):
        state = env.reset()
        for j in xrange(STEP):
          # env.render()
          action = agent.action(state) # direct action for test
          state,reward,done,_ = env.step(action)
          total_reward += reward
          if done:
            break
      ave_reward = total_reward/TEST
      # agent.set_performance(ave_reward)
      print 'episode: ',episode,'Evaluation Average Reward:',ave_reward
      if ave_reward >= 200:
        break
  # gym.upload('/tmp/cartpole-experiment-0', api_key=API_KEY)

if __name__ == '__main__':
  main()