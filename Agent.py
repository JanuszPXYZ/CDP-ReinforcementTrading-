import random
import numpy as np
import tensorflow as tf
from collections import namedtuple, deque
from tensorflow.keras.models import load_model

class Agent():
    def __init__(self, state_size, is_eval = False, model_name = ""):
        self.state_size = state_size
        self.action_size = 3
        self.memory = deque(maxlen = 1000)
        self.inventory = []
        self.model_name = model_name
        self.is_eval = is_eval

        self.gamma = 0.95 # Discount factor (how much we want our agent to 'consider'
        # future rewards)
        self.epsilon = 1
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

        self.model = load_model("models/" + model_name) if is_eval else self._model()


    def _model(self):
        '''
        Model takes in the state of the environment and returns a Q-value table
        or policy that refers to the probability distribution over actions
        '''
        model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(64, input_dim = self.state_size, activation = 'relu'),
            tf.keras.layers.Dense(32, activation = 'relu'),
            tf.keras.layers.Dense(8, activation = 'relu'),
            tf.keras.layers.Dense(self.action_size, activation = 'linear')
        ])

        model.compile(loss = 'mse', optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001))

        return model


    def act(self, state):
        '''
        This function returns an ACTION given STATE. We use the model that we defined previously
        to calculate the probability distribution over actions and then we use the argmax function
        to select the action.
        '''

        if not self.is_eval and random.random() <= self.epsilon:
            return random.randrange(self.action_size)

        options = self.model.predict(state)

        return np.argmax(options[0])

    def expReplay(self, batch_size):
        '''
        The gist of the whole class, so to speak. This method is responsible for training the neural network
        based on the observed experience.
        '''
        mini_batch = []
        l = len(self.memory)
        
        # Step 1) Prepare replay memory.
        for i in range(l - batch_size + 1, l):
            mini_batch.append(self.memory[i])
        
        # Step 2) Loop across the replay memory batch.
        for state, action, reward, next_state, done in mini_batch:
            target = reward # reward or Q at time t
            
            # Step 3) Update the target for Q-tabaale
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
                
            # Step 4) Q-value of the current state in the table
            target_f = self.model.predict(state)
            
            # Step 5) Update the output Q-table for the given action in the table
            target_f[0][action] = target
            
            # Step 6) Train and fit the model.
            self.model.fit(state, target_f, epochs = 1, verbose = 0)
            
        # Step 7) Implement epsilon-greedy algorithm
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
