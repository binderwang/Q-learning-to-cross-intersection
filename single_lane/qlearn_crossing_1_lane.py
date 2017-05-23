# -*- coding: utf-8 -*-

__author__  = 'télémaque (David Forester)'
__version__ = '1.0.0'
__date__    = '20 May 2017'

'''
Use Q-learning to train a self-driving-vehicle (sdv) to a busy lane of traffic.

usage: qlearn_crossing_2_lanes.py [-h] [-s] [-c]

optional arguments:
  -h, --help        show this help message and exit
  -s, --show        show the world grid
  -c, --continuing  continue training the saved model

OUTPUT: model.h5  & model.json
'''

import os
import argparse
import json
import random

# suppress the TensorFlow warning
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import numpy as  np
from scipy.ndimage.interpolation import shift
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.optimizers import sgd

import custom_plots


class Drive(object):
    
    def __init__(self, grid_dims):
        self.grid_dims = grid_dims
        self.reset()

    def _update(self, action):
        '''
        determine whether crossed successfully or crashed
        '''
        if (self.world[7,10]==self.car_val) and (self.vert_road[7]==self.sdv_val):
            self.crashed = True
        if np.sum(self.vert_road[4:7])==self.sdv_val*3:
            self.crossed = True
            
    def _get_reward(self):
        if self.crossed:
            reward = 5
        elif self.crashed:
            reward = -10
        else:
            reward = 0
        return reward

    def _is_over(self):
        if self.crashed or self.crossed:
            return True
        else:
            return False

    def _draw_image(self):
        '''
        combine a copy of self.world with the vertical road
        '''
        image = self.world.copy()
        crossing_val = max(image[7,10],self.vert_road[7])
        image[:,10] = self.vert_road
        image[7,10] = crossing_val
        return image

    def observe(self):
        image = self._draw_image()
        return image.reshape((1, -1))

    def at_intersection(self):
        if self.y_sdv == 8:
            return True
        else:
            return False

    def crossing_opportunity(self):
        test_array = np.ones([3], np.uint8)*self.road_val
        gap_array = self.world[7,10:13]
        if np.array_equal(test_array, gap_array):
            return True
        else:
            return False
            
    def display(self):
        image = self._draw_image()
        return image

    def propagate_horz(self):
        '''
        move forward in time by one step (traffic moves, sdv moves or stays)
        '''
        if self.car_emerging:
            self.world[7,:] = shift(self.world[7,:], -1, cval=self.car_val)
        else:
            self.world[7,:] = shift(self.world[7,:], -1, cval=self.road_val)

        if self.car_emerging:
            self.world[7,-1] = self.car_val

        self.car_emerging_list.append(self.car_emerging)
        self.car_emerging_list = self.car_emerging_list[-3:]

        if not self.car_emerging:
            # 1-in-car_length*2 chance that car_emerging will become True again
            max_int = 1*self.car_length
            chance = random.randint(1, max_int)
            if chance == max_int:
                self.car_emerging = True

        if all(self.car_emerging_list):
            self.car_emerging = False

    def propagate_vert(self, action):
        '''
        move the self-driving-vehicle (sdv) forward or stay in same location
        '''
        if action == 1: # forawrd
            if self.sdv_emerging:
                self.vert_road =  shift(self.vert_road, -1, cval=self.sdv_val)
                if np.sum(self.vert_road[-4:]) == self.sdv_val*4:
                    self.vert_road[-1] = self.road_val
                    self.sdv_emerging = False
            else:
                self.vert_road =  shift(self.vert_road, -1, cval=self.road_val)
            self.y_sdv -= 1
        self._update(action)
        reward = self._get_reward()
        trial_over = self._is_over()
        return trial_over, reward, self.observe()

    def reset(self):
        '''
        initialize the environment
        '''
        self.car_emerging = True
        self.sdv_emerging = True
        self.crashed = False
        self.crossed = False
        self.car_length = 3
        self.car_emerging_list = [self.car_length*False]
        self.y_sdv = self.grid_dims[0]
        self.road_val = 1
        self.car_val = 2
        self.sdv_val = 3

        #create the grid and roads
        self.world = np.zeros(self.grid_dims, np.uint8)
        self.vert_road = np.ones(self.grid_dims[0], np.uint8)*self.road_val

        # initialize the cars on horizontal road
        self.world[7,:] = self.road_val # horizontal road our car must cross

        # initialize the sdv on the vertical road
        self.world[:,10] = self.road_val
        self.world[-1,10] = self.car_val

        # propagate forward to add cars to horizontal road
        for ii in range(self.grid_dims[1]):
            self.propagate_horz()


class ExperienceReplay(object):
    def __init__(self, max_memory=100, discount=.9):
        self.max_memory = max_memory
        self.memory = list()
        self.discount = discount

    def remember(self, states, trial_over):
        self.memory.append([states, trial_over])
        if len(self.memory) > self.max_memory:
            del self.memory[0]

    def get_batch(self, model, batch_size=10):
        len_memory = len(self.memory)
        num_actions = model.output_shape[-1]
        env_dim = self.memory[0][0][0].shape[1]
        inputs = np.zeros((min(len_memory, batch_size), env_dim))
        targets = np.zeros((inputs.shape[0], num_actions))
        for i, idx in enumerate(np.random.randint(0, len_memory,
                                                  size=inputs.shape[0])):
            state_t, action_t, reward_t, state_tp1 = self.memory[idx][0]
            trial_over = self.memory[idx][1]

            inputs[i:i+1] = state_t[0]
            targets[i] = model.predict(state_t)[0]
            Q_sa = np.max(model.predict(state_tp1)[0])
            if trial_over:
                targets[i, action_t] = reward_t
            else:
                # reward_t + gamma * max_a' Q(s', a')
                targets[i, action_t] = reward_t + self.discount * Q_sa
        return inputs, targets


if __name__ == "__main__":
    
    # Argument parser ------------------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--show', action='store_true',
                        help='show the world grid')
    parser.add_argument('-c', '--continuing', action='store_true',
                        help='continue training the saved model')                        
    args = parser.parse_args()    
    
    # parameters
    grid_dims = (21,21) # the size of the observable world
    epsilon = .3  # exploration
    num_actions = 2  # [move_forward, stay]
    epoch = 500
    max_memory = 500
    hidden_size = grid_dims[0]*grid_dims[1]
    batch_size = 50

    model = Sequential()
    model.add(Dense(hidden_size, input_shape=(grid_dims[0]*grid_dims[1],), activation='relu'))
    model.add(Dense(hidden_size, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(hidden_size, activation='relu'))
    model.add(Dense(num_actions))
    model.compile(optimizer='rmsprop',loss='mse')
    
    # To continue training from a previous model
    if args.continuing:
        print('continuing training of saved model...')
        model.load_weights('model.h5')

    # Define environment
    env = Drive(grid_dims)

    # Initialize experience replay object
    exp_replay = ExperienceReplay(max_memory=max_memory)

    # Train
    win_cnt = 0
    for e in range(epoch):
        loss = 0.
        env.reset()
        trial_over = False
        # get initial input
        input_t = env.observe()

        while not trial_over:
            input_tm1 = input_t

            # update the state of traffic on the road to cross
            env.propagate_horz()

            # update the postition of our sdv
            at_intersection = env.at_intersection()
            if not at_intersection:
                action = 1
            else:
                if np.random.rand() <= epsilon:
                    # explore
                    action = np.random.randint(0, num_actions, size=1)[0]
                else:
                    # follow policy
                    q = model.predict(input_tm1)
                    action = np.argmax(q[0])

            trial_over, reward, input_t = env.propagate_vert(action)
            if reward > 0:
                win_cnt += 1

            # store experience
            exp_replay.remember([input_tm1, action, reward, input_t], trial_over)

            # adapt model
            inputs, targets = exp_replay.get_batch(model, batch_size=batch_size)

            # show the world
            if args.show:
                ans = custom_plots.show_img_return_input(env.display(), 'Driving', ask=True)

            loss += model.train_on_batch(inputs, targets)#[0]
        print("Epoch {:03d}/{:d} | Loss {:.4f} | Success count {}".format(e, epoch, loss, win_cnt))

    # Save trained model weights and architecture
    model.save_weights("model.h5", overwrite=True)
    with open("model.json", "w") as outfile:
        json.dump(model.to_json(), outfile)