import numpy as np


class Memory:
    def __init__(self, max_size):
        self.buffer = []
        # list of booleans to identify the moves with positive rewards to prioritize their sampling
        self.positives = []
        # add max_size to update the memory by removing the old experiments (where the model was less accurate)
        self.max_size = max_size

    def add(self, experience, custom_rewards):
        """experience [current_state, action, reward, new_state, env.game_over()]"""
        if len(self.buffer) == self.max_size:
            # memory is of limited space so we remove the oldest examples before adding new ones
            if sum(self.positives) < 0.2*self.max_size and self.positives[0]:
                # if not enough positive examples, we try not to remove one
                # we randomly swap with a negative sample that will be removed instead
                index = np.random.randint(self.max_size)
                while self.positives[index]:
                    index = np.random.randint(self.max_size)
                self.buffer[0], self.buffer[index] = self.buffer[index], self.buffer[0]
                self.positives[0], self.positives[index] = self.positives[index], self.positives[0]

            self.buffer = self.buffer[1:]
            self.positives = self.positives[1:]
        self.buffer += [experience]
        self.positives += [experience[2] > 20*custom_rewards]

    def sample(self, batch_size):
        """choose batch_size elements among the memory of the agent
        with at least batch_size//3 elements with a positive reward to ensure that the training is valuable ???
        """
        buffer_size = len(self.buffer)
        num_pos = sum(self.positives)

        # if very few positive rewards we select them all
        if num_pos == 0:
            index = list(np.random.choice(np.arange(buffer_size),
                                  size=batch_size,
                                  replace=False))
        elif num_pos < batch_size/3:
            print('too few positive examples')
            index = list(np.nonzero(self.positives)[0]) + list(np.random.choice(np.arange(buffer_size),
                                 size=batch_size - num_pos,
                                 replace=False))
        else:
            index = list(np.random.choice(np.nonzero(self.positives)[0],
                                 size=batch_size//3,
                                 replace=False)) \
                    + list(np.random.choice(np.arange(buffer_size),
                                 size=batch_size - batch_size//3,
                                 replace=False))

        return [self.buffer[i] for i in index]

