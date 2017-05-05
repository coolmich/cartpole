### naive_dqn.py
* deep Q-learning with only replay memory
* unstable, typically needs a lot more than 1k episodes to train
* tried to adjust epsilon according to discrepency between the target score of 200 and current reward, but the high variance of training rewards prevents meaningful comparison, no improvement could be noticed

### double_dqn.py
* DQN with both replay memory and a target network that implements fixed Q update(Double DQN paper)
* relatively more stable, typically needs around 1k episodes to converge