# M4 - Autonomous fruit searching

# basic python packages
# from cmath import pi
from pickle import FALSE
import sys, os
from time import time
# import cv2
import numpy as np
import json
import ast
import argparse
import time
# from time import sleep

# import utility functions
sys.path.insert(0, "util")
from pibot import Alphabot
import measure as measure

# -------------------------------- Newly added imports --------------------------------
from CJS_operate import Operate
import pygame

# -------------------------------- Newly added imports --------------------------------



def read_true_map(fname):
    """Read the ground truth map and output the pose of the ArUco markers and 3 types of target fruit to search

    @param fname: filename of the map
    @return:
        1) list of target fruits, e.g. ['redapple', 'greenapple', 'orange']
        2) locations of the target fruits, [[x1, y1], ..... [xn, yn]]
        3) locations of ArUco markers in order, i.e. pos[9, :] = position of the aruco10_0 marker
    """
    with open(fname, 'r') as f:
        try:
            gt_dict = json.load(f)                   
        except ValueError as e:
            with open(fname, 'r') as f:
                gt_dict = ast.literal_eval(f.readline())   
        fruit_list = []
        fruit_true_pos = []
        aruco_true_pos = np.empty([10, 2])
        print(gt_dict)

        # remove unique id of targets of the same type
        for key in gt_dict:
            x = np.round(gt_dict[key]['x'], 1)
            y = np.round(gt_dict[key]['y'], 1)

            if key.startswith('aruco'):
                if key.startswith('aruco10'):
                    aruco_true_pos[9][0] = x
                    aruco_true_pos[9][1] = y
                else:
                    marker_id = int(key[5])
                    aruco_true_pos[marker_id-1][0] = x
                    aruco_true_pos[marker_id-1][1] = y
            else:
                fruit_list.append(key[:-2])
                if len(fruit_true_pos) == 0:
                    fruit_true_pos = np.array([[x, y]])
                else:
                    fruit_true_pos = np.append(fruit_true_pos, [[x, y]], axis=0)

        return fruit_list, fruit_true_pos, aruco_true_pos



def read_search_list():
    """Read the search order of the target fruits

    @return: search order of the target fruits
    """
    search_list = []
    with open('search_list.txt', 'r') as fd:
        fruits = fd.readlines()

        for fruit in fruits:
            search_list.append(fruit.strip())

    return search_list



def print_target_fruits_pos(search_list, fruit_list, fruit_true_pos):
    """Print out the target fruits' pos in the search order

    @param search_list: search order of the fruits
    @param fruit_list: list of target fruits
    @param fruit_true_pos: positions of the target fruits
    """

    print("Search order:")
    n_fruit = 1
    for fruit in search_list:
        for i in range(5):
            if fruit == fruit_list[i]:
                print('{}) {} at [{}, {}]'.format(n_fruit,
                                                  fruit,
                                                  np.round(fruit_true_pos[i][0], 1),
                                                  np.round(fruit_true_pos[i][1], 1)))
        n_fruit += 1



# def drive_take_pic_update_slam(operate, command=[0,0], tick=0.0, turning_tick=0.0, time=0.0):
#     # operate.take_pic()
#     sleep(0.5)
#     lv, rv = operate.pibot.set_velocity(command, tick=tick, turning_tick=turning_tick, time=time)
#     sleep(0.5)
#     operate.take_pic()
#     drive_meas = measure.Drive(lv, rv, time)
#     operate.update_slam(drive_meas)
#     operate.draw(canvas)
#     pygame.display.update()
#     # operate.notification = f'{operate.ekf.get_state_vector()[0:3, :]}'







# quit GUI when user clicks ESC
def check_to_quit_gui():
    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            # stop robot in case it's still moving
            operate.command['motion'] = [0,0]
            _ = operate.control()
            # close GUI
            pygame.quit()
            sys.exit()
        



if __name__ == "__main__":
    parser = argparse.ArgumentParser("Fruit searching")
    parser.add_argument("--map", type=str, default='../M3/EST_TRUEMAP.txt')
    parser.add_argument("--ip", metavar='', type=str, default='localhost')
    parser.add_argument("--port", metavar='', type=int, default=8000)
    parser.add_argument("--calib_dir", type=str, default="calibration/param/")
    parser.add_argument("--save_data", action='store_true')
    # parser.add_argument("--play_data", action='store_true')
    parser.add_argument("--ckpt", default='network/scripts/model/model.best.pth')
    args, _ = parser.parse_known_args()

    # debugging parameter
    plot_djikstra_path = True

    ppi = Alphabot(args.ip, args.port)

    # read in the true map
    fruits_list, fruits_true_pos, aruco_true_pos = read_true_map(args.map)
    search_list = read_search_list()
    print_target_fruits_pos(search_list, fruits_list, fruits_true_pos)

    waypoint = [0.0,0.0]
    robot_pose = [0.0,0.0,0.0]




    # initialize GUI
    pygame.font.init()
    width, height = 700, 660
    canvas = pygame.display.set_mode((width, height))
    pygame.display.set_caption('ECE4078 2021 Lab')
    pygame.display.set_icon(pygame.image.load('pics/8bit/pibot5.png'))
    canvas.fill((0, 0, 0))
    splash = pygame.image.load('pics/loading.png')
    pibot_animate = [pygame.image.load('pics/8bit/pibot1.png'),
                    pygame.image.load('pics/8bit/pibot2.png'),
                    pygame.image.load('pics/8bit/pibot3.png'),
                    pygame.image.load('pics/8bit/pibot4.png'),
                    pygame.image.load('pics/8bit/pibot5.png')]
    pygame.display.update()

    # prompt user to start GUI
    start_gui = False
    counter = 40
    while not start_gui:
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                start_gui = True
        canvas.blit(splash, (0, 0))
        x_ = min(counter, 600)
        if x_ < 600:
            canvas.blit(pibot_animate[counter%10//2], (x_, 565))
            pygame.display.update()
            counter += 2

    operate = Operate(args)
    operate.notification = 'Press ENTER to start SLAM'

    ##### run SLAM #####
    # operate.ekf_on = True
    # if operate.ekf_on:
    #         operate.notification = 'SLAM is running'
    # n_observed_markers = len(operate.ekf.taglist)
    # if n_observed_markers == 0:
    #     if not operate.ekf_on:
    #         print('SLAM is running')
    #         operate.ekf_on = True
    #     else:
    #         print('> 2 landmarks is required for pausing - 1st')
    # elif n_observed_markers < 3:
    #     print('> 2 landmarks is required for pausing')
    # else:
    #     if not operate.ekf_on:
    #         operate.request_recover_robot = True
    #     operate.ekf_on = not operate.ekf_on
    #     if operate.ekf_on:
    #         print('SLAM is running')
    #     else:
    #         print('SLAM is paused')

    ##### Initialise SLAM states with marker position from true map #####
    operate.ekf.freeze_markers = True
    LMS = []
    for idx, marker in enumerate(aruco_true_pos):
        LMS.append(measure.Marker(np.expand_dims(aruco_true_pos[idx], -1), idx+1))
    operate.ekf.add_landmarks(LMS)
    operate.draw(canvas)
    pygame.display.update()

    start = False
    while not start:
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN:
                start = True
                operate.ekf_on = True
                if operate.ekf_on == True:
                    operate.notification = 'SLAM is running'
        operate.take_pic()
        drive_meas = operate.control()
        # operate.update_slam(drive_meas)
        operate.draw(canvas)
        pygame.display.update()
    

    calibration_on = False
    visited_fruit_count = 0    
    while start_gui:        
        while visited_fruit_count < 3:
            if calibration_on:
                print('calibration mode')
                # calibrate forward motion
                # operate.forward_calibrator()

                # CCW
                # operate.drive_take_pic_update_slam(command=[0,1], turning_tick=20, time=10)

                # CW
                # operate.drive_take_pic_update_slam(command=[0,-1], turning_tick=20, time=10)

                # operate.turning_calibrator(turning_angle=45, turning_count=8)
                # operate.turning_calibrator(turning_angle=135, turning_count=4)
                # operate.turning_calibrator(turning_angle=90, turning_count=2)

                # operate.turning_calibrator(turning_angle=45, turning_count=8, mode='CW')
                # operate.turning_calibrator(turning_angle=90, mode='CW')
                # operate.turning_calibrator(turning_angle=135, mode='CW', turning_count=4)
                # operate.turning_calibrator(turning_angle=180, mode='CW')

                # to exit while loop
                visited_fruit_count = 3
                start_gui = False

            # pygame.time.Clock().tick(60)            

            ##### test max and min turn vel #####
            # operate.take_pic()
            # operate.command['motion'] = [0,1]
            # drive_meas = operate.control(turning_tick=15)
            # operate.update_slam(drive_meas)
            # operate.draw(canvas)
            # pygame.display.update()
            ##### test max and min turn vel #####

            

            # LOOP over each target fruit
            for fruit in search_list:
                # target_fruit_idx = 0
                for i in range(5):
                    if fruit == fruits_list[i]:
                        target_fruit_idx = i
                        operate.notification = f'Visiting fruit {visited_fruit_count+1}: {fruit}'
                
                # create occupancy map
                obs_x, obs_y = operate.create_occupancy_map(fruits_true_pos, aruco_true_pos, target_fruit_idx)


                # Dijkstra path planning algorithm
                waypts_x, waypts_y = operate.run_Djikstra(fruits_true_pos, target_fruit_idx, obs_x, obs_y, plot_path=plot_djikstra_path)
                
                # print('goal', waypts_x[0], waypts_y[0])
                # LOOP over each waypoints
                x, y = 0.0, 0.0
                for i in range(1, len(waypts_x)-1): # loop the navigation waypoint
                # for i in range(1): # loop the navigation waypoint
                    x = waypts_x[-i-1]
                    y = waypts_y[-i-1]
                    waypoint = [x,y]
                    # print('waypoint: ', waypoint)
                    # waypoint = [0.4, 0.4]
                    # waypoint = [0.4, -0.4]
                    # waypoint = [-0.4, 0.0]  # 180 CCW
                    # waypoint = [0.04, 0.4]  # 90 CCW
                    # print('goal', waypts_x[0], waypts_y[0])

                    ##### turn towards waypoint #####
                    operate.turn_to_waypoint(waypoint)
                    # print(operate.get_robot_pose())
                    operate.update_gui(canvas)

                    operate.go_to_waypoint(waypoint)
                    print(operate.get_robot_pose())
                    operate.update_gui(canvas)

                    # user = input('Y/N')

                    
                    # every 3 moves, rotate to look for aruco markers
                    if i % 3 == 0:
                        for _ in range(8):
                            operate.look_for_ARUCO()
                            operate.update_gui(canvas)

                    
                    if i == len(waypts_x) - 2:  # reached target fruit
                        print('Reached target')
                        visited_fruit_count += 1
                        operate.notification = f'Reached goal!'

                        # only scan if robot not at last target fruit
                        if visited_fruit_count < 3:
                            for _ in range(8):
                                operate.look_for_ARUCO()
                                operate.update_gui(canvas)
                        
                        # user = input('Y/N')
                        # if user=='Y':
                        #     continue
            
            operate.notification = f'Visited all target fruits!'

        operate.draw(canvas)
        pygame.display.update()
        check_to_quit_gui()
        