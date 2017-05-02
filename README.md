### naive_dqn.py
* deep Q-learning with only replay memory
* unstable, typically needs more than 1k episodes to train
* tried to adjust epsilon according to discrepency between the target score of 200 and current reward, but the high variance of training rewards prevents meaningful comparison, no improvement could be noticed