from ple.games.snake import Snake
from ple import PLE
from agent import Agent
from utils import *

#go left p.act
# import gym
# from PIL import Image
#
# env = gym.make('Pong-v0')
# env.reset()
# for _ in range(1000):
#     env.step(env.action_space.sample())
#     env.render('human')
# env.close()

NB_ACTIONS = 4
EXPLORATION_RATE = 0.5
DISCOUNT = 0.95
MAX_DURATION = 1000
MINI_BATCH_SIZE = 16
MEMORY_MAX_SIZE = 10000
DECAY = 0.999
INITIALIZATION_SIZE = 2*MINI_BATCH_SIZE
NUM_EPISODES = 3000


def main():
    game = Snake()
    game = PLE(game, display_screen=True)
    game.init()
    action_space = game.getActionSet()
    game.act(0)
    agent = Agent(game, MEMORY_MAX_SIZE, EXPLORATION_RATE, DISCOUNT, INITIALIZATION_SIZE)
    return game, agent


env, agent = main()
agent.train(NUM_EPISODES)
