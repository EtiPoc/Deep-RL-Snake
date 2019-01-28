from memory import Memory
from model import Model
from utils import *
import numpy as np

MINI_BATCH_SIZE = 16
DECAY = 0.999


class Agent:
    def __init__(self, env, memory_max_size, exploration_rate, discount_factor, initialization_size, custom_rewards=True):
        self.env = env
        self.memory = Memory(memory_max_size)
        self.qnet = Model()
        self.num_episodes = 0
        self.exploration_rate = exploration_rate
        self.training = True
        self.loss_history = []
        self.rewards_history = []
        self.discount_factor = discount_factor
        self.custom_rewards = custom_rewards
        self.initialize_memory(initialization_size)

    def initialize_memory(self, initialization_size, nb_actions=4):
        """fill the buffer with 2*batch_size of random actions """
        action_space = self.env.getActionSet()
        current_state = self.env.getScreenGrayscale() - 25
        current_state = current_state.reshape((1, 64, 64, 1))
        # no preprocessing required on this image
        num_actions = 1

        while num_actions < initialization_size:
            # choose action considering the exploration rate
            if self.env.game_over():
                self.env.reset()
            action = np.random.randint(nb_actions)
            reward = 5 * self.env.act(action_space[action])

            num_actions += 1
            new_state = (self.env.getScreenGrayscale() - 25).reshape((1, 64, 64, 1))
            self.memory.add([current_state, action, reward, new_state, self.env.game_over()], self.custom_rewards)
            current_state = new_state

    def process_batch(self, batch):
        """
        takes the sampled element in the memory
        compute the target for each one of them (the predicted Q-value for the actions not taken
        and the target for the taken action)
        :param batch:
        :return:
        """
        targets = []
        states = []
        for sample in batch:
            states += [sample[0].reshape((64, 64, 1))]
            target = self.qnet.model.predict(sample[0])[0]

            # if next state not terminal
            if not sample[-1]:
                # target of the action taken is the reward plus the discounted q-value of the next state
                target[sample[1]] = sample[2] + self.discount_factor * np.max(self.qnet.model.predict(sample[3]))
            # else the target is only the reward
            else:
                # target is the reward (as the state is terminal there shall be no next state
                target[sample[1]] = sample[2]
            targets += [target]
        return states, targets

    def act(self, state):
        # if not training mode, chose the best possible action (ie: with highest Q-value)
        if not self.training:
            action = np.argmax(self.qnet.model.predict(state))
        else:
            sample = np.random.rand()
            if sample < self.exploration_rate:
                action = np.random.randint(4)
            else:
                # chose the action argmax of the Q-function for the current state
                action = np.argmax(self.qnet.model.predict(state))

        return action

    def train(self, episodes):
        action_space = self.env.getActionSet()
        num_pos = []
        share_pos = []
        for episode in range(1, episodes):
            # initialization
            self.num_episodes += 1
            self.env.init()
            episode_reward = 0
            current_state = self.env.getScreenGrayscale() - 25
            current_state = current_state.reshape((1, 64, 64, 1))
            # no preprocessing required on this image
            num_actions = 1
            print('"""PLAYING EPISODE %s"""' % episode)

            # max duration of 1000 actions
            while num_actions < 1000 and not self.env.game_over():
                # choose action considering the exploration rate
                action = self.act(current_state)
                reward = self.env.act(action_space[action])

                if self.env.game_over():
                    reward *= 2
                if reward == 1:
                    reward *= 100
                    print('"""ate something"""')
                if reward == 0 and self.custom_rewards:
                    reward = custom_reward(self.env)

                num_actions += 1
                episode_reward += reward
                new_state = (self.env.getScreenGrayscale() - 25).reshape((1, 64, 64, 1))
                self.memory.add([current_state, action, reward, new_state, self.env.game_over()], self.custom_rewards)
                current_state = new_state

    # sample batch from the memory and train the model
                training_samples = self.memory.sample(MINI_BATCH_SIZE)
                training_samples, training_targets = self.process_batch(training_samples)

    # score updates and plotting
                num_pos += [sum(self.memory.positives)]
                share_pos += [sum(self.memory.positives) / self.memory.buffer.__len__()]
                self.loss_history += [
                    self.qnet.model.train_on_batch(np.array(training_samples), np.array(training_targets))]
            print('"""EPISODE REWARD : %s"""' % episode_reward)
            self.rewards_history += [episode_reward]

            if episode % 100 == 0:
                print('reward at episode %s = %s' % (episode, np.mean(self.rewards_history[-100:])))
                print('loss at episode %s = %s' % (episode, np.mean(self.loss_history[-100:])))
                plot4(episode, self.loss_history[-100:], self.rewards_history[-100:], num_pos[-100:],
                      share_pos[-100:])
                self.exploration_rate *= DECAY
        plot4(self.num_episodes, self.loss_history, self.rewards_history, num_pos, share_pos)

