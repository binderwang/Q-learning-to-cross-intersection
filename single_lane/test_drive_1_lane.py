# -*- coding: utf-8 -*-

__author__  = 'télémaque (David Forester)'
__version__ = '1.0.0'
__date__    = '20 May 2017'

'''
Perform trials of the simulated vehicle crossing a busy lane of traffic using
the pre-trained model (trained with qlearn_crossing_1_lane.py).

usage: test_drive_1_lane.py [-h] [-s] [-w]

optional arguments:
  -h, --help   show this help message and exit
  -s, --show   show the world grid
  -w, --write  write the state images to file (requires that avconv be installed)
'''

import os
import argparse
import json

# suppress the TensorFlow warning
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import numpy as np
from keras.models import model_from_json

from qlearn_crossing_1_lane import Drive
import custom_plots


if __name__ == "__main__":
    # Make sure this grid size matches the value used fro training

    with open("model.json", "r") as jfile:
        model = model_from_json(json.load(jfile))
    model.load_weights("model.h5")
    model.compile(optimizer='rmsprop',loss='mse')
    
    # Argument parser ------------------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--show', action='store_true',
                        help='show the world grid')
    parser.add_argument('-w', '--write', action='store_true',
                        help='write the environment state images as PNGs')                   
    args = parser.parse_args()
    
    out_dir = './output'
    
    # Define environment, game
    grid_dims = (21,21) # the size of the observable world
    env = Drive(grid_dims)
    numb_tests = 1000
    win_cnt = 0 # number of successful crossings
    opp_cnt = 0 # number of crossing opportunities
    ops_missed = 0 # number of crossing opportunities forgone
    itest = 0
    for e in range(numb_tests):
        loss = 0.
        env.reset()
        trial_over = False
        # get initial input
        input_t = env.observe()
        iframe = 0
        while not trial_over:
            input_tm1 = input_t
            # update the state of traffic on the road to cross
            env.propagate_horz()
            
            # check if a crossing opportunity exists (at least 3 pixels wide)
            should_cross = env.crossing_opportunity()
            if should_cross:
                opp_cnt += 1

            # update the postition of our sdv
            at_intersection = env.at_intersection()
            if not at_intersection:
                action = 1
            else:
                q = model.predict(input_tm1)
                action = np.argmax(q[0])
                #action = 1

            if should_cross and action==0:
                ops_missed += 1

            # apply action, get reward
            trial_over, reward, input_t = env.propagate_vert(action)
            if reward > 0:
                win_cnt += 1

            # either write or display frames
            if args.write:
                # write numbered frames to output directory
                itn = str(itest).zfill(3)
                ifn = str(iframe).zfill(2)
                title = 'trial-'+itn+'_frame-'+ifn
                custom_plots.write_img(env.display(), title, out_dir)
            else:
                # display the world grid
                if args.show:
                    ans = custom_plots.show_img_return_input(env.display(),
                                                             'Testing Policy',
                                                             ask=False)
                
            iframe += 1
                                                         
        #print("Test # {:03d} | Success count {}".format(e, win_cnt))
        itest += 1
        
    print('\nMissed Opportunites:{0:.2f} %'.format(100*ops_missed/opp_cnt))
    print('\nSuccess Rate: {0:.2f} %'.format(100*win_cnt/numb_tests))                                                  
