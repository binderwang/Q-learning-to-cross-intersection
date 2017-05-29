import os
import argparse
import json

# suppress the TensorFlow warning
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import numpy as np
from keras.models import model_from_json

from qlearn_crossing import Drive
import custom_plots


l1_row = 11
l2_row = 10
l3_row = 5
l4_row = 4     
l1_model_row = 8
l2_model_row = 7
road_val = 1
car_val = 2
sdv_val = 3  
def convert_to_2_lanes(four_lane_image, intersection_number):
    intersection1 = four_lane_image[l2_row:l1_row+1, :]
    intersection2 = four_lane_image[l4_row:l3_row+1, :]
    # empty world grid
    img = np.zeros((21,21), np.uint8)
    # add vertical road
    img[:, 10] = road_val
    if intersection_number == 1:
        # add the intersection
        img[l2_model_row:l1_model_row+1,:] = intersection1
    elif intersection_number == 2:
        # add the intersection
        img[l2_model_row:l1_model_row+1,:] = intersection2
    # add the car at the intersection
    img[l1_model_row + 1:l1_model_row + 4, 10] = sdv_val
    return img.reshape((1, -1))
     

if __name__ == "__main__":
    # Make sure this grid size matches the value used fro training
    print('Loading the two-lane version trained model...')
    with open("../two_lane_version/model.json", "r") as jfile:
        model = model_from_json(json.load(jfile))
    model.load_weights("../two_lane_version/model.h5")
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
    numb_tests = 3
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
        intersection_number = 1
        while not trial_over:
            input_tm1 = input_t
            # update the state of traffic on the road to cross
            env.propagate_horz()

            # choose and take the sdv action
            intersection_number = env.at_intersection()
            if intersection_number == 0:
                action = 1
            else:
                crossing_opportunity = env.crossing_opportunity(intersection_number)
                four_lane_image = input_tm1.reshape(grid_dims[0],grid_dims[1])
                input_tm1 = convert_to_2_lanes(four_lane_image, intersection_number)
                q = model.predict(input_tm1)
                action = np.argmax(q[0])
                if crossing_opportunity:
                    opp_cnt += 1
                if crossing_opportunity and action == 0:
                    ops_missed += 1

            passed_first_intersection = env.passed_first_intersection()
            #print(passed_first_intersection)
            if passed_first_intersection:
                intersection_number = 2
            
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
        
    print('\nMissed Opportunities:{0:.2f} %'.format(100*ops_missed/opp_cnt))
    print('\nSuccess Rate: {0:.2f} %'.format(100*win_cnt/numb_tests))                                                  
