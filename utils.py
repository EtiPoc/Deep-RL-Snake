import matplotlib.pyplot as plt
import numpy as np


def plot4(episode, losses, rewards, num_pos, share_pos):
    plt.subplot(4, 1, 1)
    plt.plot(losses)
    plt.ylabel("Losses")
    plt.subplot(4, 1, 2)
    plt.plot(rewards)
    plt.ylabel('Rewards')
    plt.subplot(4, 1, 3)
    plt.plot(num_pos)
    plt.ylabel('positive rewards in the memory')
    plt.subplot(4, 1, 4)
    plt.plot(share_pos)
    plt.ylabel('positive rewards share in the memory')
    plt.savefig('plots/scores_episode_%s_.jpg' % episode)
    plt.close()


def dist_to_food(env):
    state = env.getGameState()
    snake = [state['snake_head_x'], state['snake_head_y']]
    food = [state['food_x'], state['food_y']]
    return np.sqrt((snake[0]-food[0])**2 +(snake[1]-food[1])**2)


def custom_reward(env):
    dist = dist_to_food(env)
    return int(np.ceil(20/dist))