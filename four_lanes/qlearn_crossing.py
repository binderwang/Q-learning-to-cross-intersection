# -*- coding: utf-8 -*-

__author__  = 'télémaque'
__version__ = '1.0.0'
__date__    = '25 May 2017'

'''
Trains a self-driving-vehicle (sdv) to cross two busy intersections using Q-learning.
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

import custom_plots


class Drive(object):
    
    def __init__(self, grid_dims):
        self.grid_dims = grid_dims
        self.reset()

    def _update(self, action):
        '''
        determine whether crossed successfully or crashed
        '''
        if ((self.world[self.l1_row,10]==self.car_val) and
           (self.vert_road[self.l1_row]==self.sdv_val)):
            self.crashed = True
        if ((self.world[self.l2_row,10]==self.car_val) and
           (self.vert_road[self.l2_row]==self.sdv_val)):
            self.crashed = True
        if ((self.world[self.l3_row,10]==self.car_val) and
           (self.vert_road[self.l3_row]==self.sdv_val)):
            self.crashed = True
        if ((self.world[self.l4_row,10]==self.car_val) and
           (self.vert_road[self.l4_row]==self.sdv_val)):
            self.crashed = True                       
                   
        # has sdv passed the row of the last lane?
        if np.sum(self.vert_road[self.l4_row-3:self.l4_row])==self.sdv_val*3:
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
        l1_crossing_val = max(image[self.l1_row,10],self.vert_road[self.l1_row])
        l2_crossing_val = max(image[self.l2_row,10],self.vert_road[self.l2_row])
        l3_crossing_val = max(image[self.l3_row,10],self.vert_road[self.l3_row])
        l4_crossing_val = max(image[self.l4_row,10],self.vert_road[self.l4_row])
        image[:,10] = self.vert_road
        image[self.l1_row,10] = l1_crossing_val
        image[self.l2_row,10] = l2_crossing_val
        image[self.l3_row,10] = l3_crossing_val
        image[self.l4_row,10] = l4_crossing_val
        return image

    def observe(self):
        image = self._draw_image()
        return image.reshape((1, -1))
            
    def at_intersection(self):
        if self.y_sdv == self.l1_row + 1:
            return 1
        elif self.y_sdv == self.l3_row + 1:
            return 2
        else:
            return 0       
            
    def passed_first_intersection(self):
        if (self.y_sdv == self.l2_row - 3):
            return True
        else:
            return False      

    def crossing_opportunity(self, intersection_number):
        test_array = np.ones([3], np.uint8)*self.road_val
        if intersection_number == 1:
            l1_gap_array = self.world[self.l1_row,10:13]
            l2_gap_array = self.world[self.l2_row,7:10]
            if (np.array_equal(test_array, l1_gap_array) and
                np.array_equal(test_array, l2_gap_array)):
                return True
        elif intersection_number == 2:
            l3_gap_array = self.world[self.l3_row,10:13]
            l4_gap_array = self.world[self.l4_row,7:10]
            if (np.array_equal(test_array, l3_gap_array) and
                np.array_equal(test_array, l4_gap_array)):
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
        if self.l1_emerging:
            self.world[self.l1_row,:] = shift(self.world[self.l1_row,:], -1, cval=self.car_val)
            self.world[self.l1_row,-1] = self.car_val
        else:
            self.world[self.l1_row,:] = shift(self.world[self.l1_row,:], -1, cval=self.road_val)
        self.l1_emerging_list.append(self.l1_emerging)
        self.l1_emerging_list = self.l1_emerging_list[-3:]

        if self.l2_emerging:
            self.world[self.l2_row,:] = shift(self.world[self.l2_row,:], 1, cval=self.car_val)
            self.world[self.l2_row,-1] = self.car_val
        else:
            self.world[self.l2_row,:] = shift(self.world[self.l2_row,:], 1, cval=self.road_val)
        self.l2_emerging_list.append(self.l2_emerging)
        self.l2_emerging_list = self.l2_emerging_list[-3:]
        
        if self.l3_emerging:
            self.world[self.l3_row,:] = shift(self.world[self.l3_row,:], -1, cval=self.car_val)
            self.world[self.l3_row,-1] = self.car_val
        else:
            self.world[self.l3_row,:] = shift(self.world[self.l3_row,:], -1, cval=self.road_val)
        self.l3_emerging_list.append(self.l3_emerging)
        self.l3_emerging_list = self.l3_emerging_list[-3:]
        
        if self.l4_emerging:
            self.world[self.l4_row,:] = shift(self.world[self.l4_row,:], 1, cval=self.car_val)
            self.world[self.l4_row,-1] = self.car_val
        else:
            self.world[self.l4_row,:] = shift(self.world[self.l4_row,:], 1, cval=self.road_val)
        self.l4_emerging_list.append(self.l4_emerging)
        self.l4_emerging_list = self.l4_emerging_list[-3:]                        

        #max_int = 2*self.car_length
        max_int = 7
        if not self.l1_emerging:
            # 1-in-car_length*2 chance that emerging will become True again
            chance = random.randint(1, max_int)
            if chance == max_int:
                self.l1_emerging = True
        if not self.l2_emerging:
            # 1-in-car_length*2 chance that emerging will become True again
            chance = random.randint(1, max_int)
            if chance == max_int:
                self.l2_emerging = True
        if not self.l3_emerging:
            # 1-in-car_length*2 chance that emerging will become True again
            chance = random.randint(1, max_int)
            if chance == max_int:
                self.l3_emerging = True
        if not self.l4_emerging:
            # 1-in-car_length*2 chance that emerging will become True again
            chance = random.randint(1, max_int)
            if chance == max_int:
                self.l4_emerging = True                                         

        if all(self.l1_emerging_list):
            self.l1_emerging = False
        if all(self.l2_emerging_list):
            self.l2_emerging = False
        if all(self.l3_emerging_list):
            self.l3_emerging = False
        if all(self.l4_emerging_list):
            self.l4_emerging = False                               

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
        
        LANE_4 ============================= row 4
        LANE_3 ============================= row 5
        
        LANE_2 ============================= row 10
        LANE_1 ============================= row 11
        '''
        self.l1_emerging = True # a car emerging in lane 1
        self.l2_emerging = True # a car emerging in lane 2
        self.l3_emerging = True # a car emerging in lane 3
        self.l4_emerging = True # a car emerging in lane 4                
        self.sdv_emerging = True
        self.crashed = False
        self.crossed = False
        self.car_length = 3
        self.l1_emerging_list = [self.car_length*False]
        self.l2_emerging_list = [self.car_length*False]
        self.l3_emerging_list = [self.car_length*False]
        self.l4_emerging_list = [self.car_length*False]               
        self.y_sdv = self.grid_dims[0]
        self.l1_row = 11
        self.l2_row = 10
        self.l3_row = 5
        self.l4_row = 4              
        self.road_val = 1
        self.car_val = 2
        self.sdv_val = 3

        #create the grid and roads
        self.world = np.zeros(self.grid_dims, np.uint8)
        self.vert_road = np.ones(self.grid_dims[0], np.uint8)*self.road_val

        # initialize the cars on horizontal roads
        self.world[self.l1_row,:] = self.road_val # horizontal road our car must cross
        self.world[self.l2_row,:] = self.road_val # horizontal road our car must cross
        self.world[self.l3_row,:] = self.road_val # horizontal road our car must cross
        self.world[self.l4_row,:] = self.road_val # horizontal road our car must cross        

        # initialize the sdv on the vertical road
        self.world[:,10] = self.road_val
        self.world[-1,10] = self.car_val

        # propagate forward to add cars to horizontal road
        for _ in range(self.grid_dims[1]):
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
    epoch = 5000
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