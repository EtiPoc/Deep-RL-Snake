from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten, BatchNormalization


class Model:
    def __init__(self, nb_actions=4):
        # snake the state is only one image because there is no speed involved
        # the max size for the kernels is 3 since it is the width of the snake and its head
        self.model = Sequential()
        # self.model.add(BatchNormalization())
        self.model.add(Conv2D(64, 3, input_shape=(64, 64, 1), activation='relu'))
        # self.model.add(BatchNormalization())
        self.model.add(Conv2D(32, 3, activation='relu'))
        self.model.add(Flatten())
        self.model.add(Dense(256, activation='relu'))
        self.model.add(Dense(nb_actions, activation='linear'))
        # output the estimated Q-value for each action
        # as we only have the target for the unique selected action, the loss will be 0 for the others
        # So at each training step only the Q for one action is updated
        # the target will look like [Q(st,0), Target, Q(st,2), Q(st,3)]
        self.model.compile(loss='mse', optimizer='adam')
