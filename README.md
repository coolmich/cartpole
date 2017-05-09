### script.py
* the script to train the agent to get 200 rewards consistently in cart pole
* import any definition of the agent from the following files to run

### naive_dqn.py
* deep Q-learning with only replay memory
* unstable, typically needs a lot more than 1k episodes to train
* tried to adjust epsilon according to discrepency between the target score of 200 and current reward, but the high variance of training rewards prevents meaningful comparison, no improvement could be noticed

### double_dqn.py
* DQN with both replay memory and a target network that implements fixed Q update(Double DQN paper)
* relatively more stable, typically needs around 1k episodes to converge, but often would require a lot of episodes
* did not observe significant improvements over the naive dqn

### policy_net.py
* Policy Gradient method implemented in tensorflow
* It takes "forever" to converge with naive reward
* Instead, we use a normalized advantage value, outperforming DQN significantly
* Converges in hundreds of episodes