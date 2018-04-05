""" ML/DL for everyone (Reinforcement Learning)
# using OpenAI GYM, import gym
# https://gym.openai.com/
# agent (action) / observation / state, reward
# Q-Table method
"""
import sys
import gym

""" 리눅스 터미널 모듈 (윈도우=없음)
import tty
import termios
"""

env = gym.make("FrozenLake-v0")
observation = env.reset()

for _ in range(10**3):
    """
    # get something? reward = False
    # finish? done = False
    # information? info = state
    """
    env.render()    #환경을 화면으로 출력한다 (state)

    # your agent here (this takes random action)
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)

class GetCh(object):
    def __call__(self):
        fd = sys.stdin.fileno()
        old_setting = termios.tcgetattr(fd)

        try:
          tty.setraw(sys.stdin.fileno())
          ch = sys.stdin.read(3)

        finally:
            termios.tcsetattr(fd, termios, TCSADRAIN, old_settings)

        return ch

inkey = GetCh()

# MACROS
LEFT= 0
DOWN = 1
RIGHT = 2
UP = 3

#Key_mapping
arrow_key = {
    '\x1b[A' : UP,
    '\x1b[B' : DOWN,
    '\x1b[C' : RIGHT,
    '\x1b[D' : LEFT }



""" Exploit vs. Exploration : (1) E-greedy method
# assume small e=0.1
# - 10 percents - Exploration
# - 90 percents - Exploit : action = argmax(Q(s,a))
# - decaying (2) E-greedy method
#   : possibility reduce repeatation is getting encreased
#       * e = 0.1 / (iter+=1)
# (3) random noise method = add randome noise to reward & action.
# (4) Final Function : Q(s,a) = reward + gamma * Q(s',a;)
# - gammer = 0.9 : nearest will get more reward.
"""

""" Determistic vs Stochastic (nondetermistic) condition
# Determistic = returned reward, condition always the same
# Stochastic = not always, like a slippery ice surface.
"""

"""
# register Frozen lake with is_slippery False
"""
# resister(
#     id='FrozenLake-v3',
#     entry_point='gym.envs.toy_text:FrozenLakeEnv',
#     kwargs={'map_name' : '4x4', 'is_slippery': False}
# )
#
# env = gym.make("FrozenLake-v3")

"""  LECTURE.05 : is_slippery = True, learn a bit from Q()
# alpha = 0.1, Learning rate
# intention = 0.9, (1-alpha)
# Q(s,a) = (1-alpha)*Q(s,a) + alpha * ( r + gamma * Q(s',a'))
#
"""

""" Q-Learning : Q-Table (4x4) : FrozenLake
# Maze 100x100 = 100x100x 4-actions = 4e4, it's OK.
# Q-Networks = pow(2, 80*80) =
"""
