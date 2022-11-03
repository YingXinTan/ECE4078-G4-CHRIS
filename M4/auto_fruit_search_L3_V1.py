# M4 - Autonomous fruit searching

# basic python packages
from cmath import pi
import sys, os
from tkinter.tix import Tree
import cv2
import numpy as np
import json
import ast
import argparse
import time

# import utility functions
sys.path.insert(0, "util")
from pibot import Alphabot
import measure as measure

# -------------------------------- Newly added imports --------------------------------
from operate import Operate
import pygame
from dijkstra import Dijkstra
import matplotlib.pyplot as plt
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
        for i in range(3):
            if fruit == fruit_list[i]:
                print('{}) {} at [{}, {}]'.format(n_fruit,
                                                  fruit,
                                                  np.round(fruit_true_pos[i][0], 1),
                                                  np.round(fruit_true_pos[i][1], 1)))
        n_fruit += 1



def take_pic_update_slam(operate, command=[0,0], tick=0.0, turning_tick=0.0, time=0.0):
    # operate.take_pic()
    lv, rv = operate.pibot.set_velocity(command, tick=tick, turning_tick=turning_tick, time=time)
    operate.take_pic()
    drive_meas = measure.Drive(lv, rv, time)
    operate.update_slam(drive_meas)
    operate.draw(canvas)
    pygame.display.update()



def get_robot_pose(operate):
    robot_pose = operate.ekf.get_state_vector()[0:3,:]
    return robot_pose



def compute_delta_theta(waypoint, robot_pose):
    # turn towards the waypoint
    theta_goal = np.arctan2(waypoint[1]-robot_pose[1], waypoint[0]-robot_pose[0])
    delta_theta = theta_goal - robot_pose[2]    
    if delta_theta < -np.pi:
        delta_theta += 2*np.pi
    elif delta_theta > np.pi:
        delta_theta -= 2*np.pi
    return delta_theta



def turning(waypoint, robot_pose, operate):
    fileS = "calibration/param/scale.txt"
    scale = np.loadtxt(fileS, delimiter=',')
    fileB = "calibration/param/baseline.txt"
    baseline = np.loadtxt(fileB, delimiter=',')

    CCW_baseline = baseline - 0.01
    # reduce to turn less
    CW_baseline = baseline + 0.01

    # print(f'Before turn: {robot_pose}')
    turn__vel = 20
    delta_theta = compute_delta_theta(waypoint, robot_pose)
    # margin = 2 * np.pi / 180
    # cnt = 0


    CCW_turn_time = float( (abs(delta_theta)*CCW_baseline) / (2*turn__vel*scale) )
    CW_turn_time = float( (abs(delta_theta)*CW_baseline) / (2*turn__vel*scale) )
    # print("Turning for {:.2f} seconds".format(turn_time))

    if delta_theta > 0:
        take_pic_update_slam(operate, command=[0,1], turning_tick=turn__vel, time=CCW_turn_time)
    elif delta_theta < 0:
        take_pic_update_slam(operate, command=[0,-1], turning_tick=turn__vel, time=CW_turn_time)    
    time.sleep(1)

    # while abs(delta_theta) > margin:
    #     turn_time = float( (abs(delta_theta)*baseline) / (2*turn__vel*scale) )
    #     # print("Turning for {:.2f} seconds".format(turn_time))

    #     if delta_theta > 0:
    #         take_pic_update_slam(operate, command=[0,1], turning_tick=turn__vel, time=turn_time)
    #     elif delta_theta < 0:
    #         take_pic_update_slam(operate, command=[0,-1], turning_tick=turn__vel, time=turn_time)
        
    #     time.sleep(1)
    #     # print(f'After turn: {get_robot_pose(operate)}')
    #     robot_pose = get_robot_pose(operate)
    #     delta_theta = compute_delta_theta(waypoint, robot_pose)
    #     cnt+=1
    # print(f'\nRandike Turn Cnt: {cnt}')



def forward(waypoint, robot_pose, operate):
    fileS = "calibration/param/scale.txt"
    # to move more, reduce scale
    scale = np.loadtxt(fileS, delimiter=',') + 0.0003

    wheel_vel = 22
    # delta_d = np.sqrt((waypoint[0]-robot_pose[0])**2 + (waypoint[1]-robot_pose[1])**2) + 0.04
    delta_d = np.sqrt((waypoint[0]-robot_pose[0])**2 + (waypoint[1]-robot_pose[1])**2)
    # margin = 0.01
    # cnt = 0

    drive_time = float( delta_d / (wheel_vel*scale) )
    take_pic_update_slam(operate, command=[1,0], tick=wheel_vel, time=drive_time)
    time.sleep(1)

    # while delta_d > margin:        
    #     drive_time = float( delta_d / (wheel_vel*scale) )
    #     take_pic_update_slam(operate, command=[1,0], tick=wheel_vel, time=drive_time)
    #     time.sleep(1)

    #     robot_pose = get_robot_pose(operate)
    #     delta_d = np.sqrt((waypoint[0]-robot_pose[0])**2 + (waypoint[1]-robot_pose[1])**2)
    #     cnt+=1
    # print(f'\nRandike Forward Cnt: {cnt}')
    
        

def drive_to_point(waypoint, robot_pose, operate): 
    print(f'Before turn: {get_robot_pose(operate)}')

    turning(waypoint, robot_pose, operate)

    robot_pose = get_robot_pose(operate)
    print(f'\nAfter turn: {robot_pose}')

    forward(waypoint, robot_pose, operate)

    print(f'\nAfter forward: {get_robot_pose(operate)}')



def create_occupancy_map(fruits_true_pos, aruco_true_pos, target_fruit_idx):
    ox, oy = [], [] # obstacle location

    # define obstacle fruit's location
    for i in range(3):
        if i == target_fruit_idx: # do not include the current target fruit as obstacle
            continue
        ox.append(fruits_true_pos[i][0])
        oy.append(fruits_true_pos[i][1])
    
    # define aruco obstacle location
    for i in range(10):
        ox.append(aruco_true_pos[i][0])
        oy.append(aruco_true_pos[i][1])
    return ox, oy



def run_Djikstra(operate, fruits_true_pos, target_fruit_idx, obs_x, obs_y, plot_path=False):
    # compute robot radius from baseline
    robot_radius = float( np.loadtxt("calibration/param/baseline.txt", delimiter=',') / 2 ) * 3.5

    # occupancy grid resolution [m]
    grid_resolution = 0.2

    # current robot pose as starting position 
    robot_pose = get_robot_pose(operate)
    start_x, start_y = float(robot_pose[0]), float(robot_pose[1])

    # target fruit position as goal
    goal_x, goal_y = fruits_true_pos[target_fruit_idx][0], fruits_true_pos[target_fruit_idx][1]

    # run Dijkstra algorithm
    dijkstra = Dijkstra(obs_x, obs_y, grid_resolution, robot_radius)
    waypts_x, waypts_y = dijkstra.planning(start_x, start_y, goal_x, goal_y) # list of x and y waypoints coordinates

    # display Djikstra's path
    if plot_path:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        # Major ticks every 0.4 m, minor ticks every resolution m
        major_ticks = np.arange(dijkstra.min_x, dijkstra.max_x+grid_resolution*2, grid_resolution*2)
        minor_ticks = np.arange(dijkstra.min_x, dijkstra.max_x+grid_resolution, grid_resolution)
        ax.set_xticks(major_ticks)
        ax.set_xticks(minor_ticks, minor=True)
        ax.set_yticks(major_ticks)
        ax.set_yticks(minor_ticks, minor=True)
        ax.grid(which='both')
        ax.grid(which='minor', alpha=0.25)
        ax.grid(which='major', alpha=0.75)
        ax.plot(obs_x, obs_y, ".k")
        ax.plot(start_x, start_y, "og")
        ax.plot(goal_x, goal_y, "xb")
        ax.plot(waypts_x, waypts_y, "-r")
        plt.pause(0.001)
        plt.show()

    return waypts_x, waypts_y


def calibrate(mode):
    if mode == 'forward':
        ##### FORWARD calibration #####
        fileS = "calibration/param/scale.txt"
        scale = np.loadtxt(fileS, delimiter=',') + 0.0003    # to move more, reduce scale
        waypoint = [0.4, 0.0]
        wheel_vel = 22
        delta_d = np.sqrt((waypoint[0]-robot_pose[0])**2 + (waypoint[1]-robot_pose[1])**2)
        drive_time = float( delta_d / (wheel_vel*scale) )
        take_pic_update_slam(operate, command=[1,0], tick=wheel_vel, time=drive_time)
        ##### FORWARD calibration #####

    elif mode == 'CW':
        ##### CW calibration #####   
        fileS = "calibration/param/scale.txt"
        fileB = "calibration/param/baseline.txt"
        scale = np.loadtxt(fileS, delimiter=',')
        baseline = np.loadtxt(fileB, delimiter=',')
        CW_baseline = baseline + 0.01   # reduce to turn less

        turn__vel = 20
        delta_theta = compute_delta_theta([0.4, -0.4], robot_pose)
        CW_turn_time = float( (abs(delta_theta)*CW_baseline) / (2*turn__vel*scale) )
        take_pic_update_slam(operate, command=[0,-1], turning_tick=turn__vel, time=CW_turn_time)
        ##### CW calibration #####

    elif mode == 'CCW':
        ##### CCW calibration #####   
        fileS = "calibration/param/scale.txt"
        fileB = "calibration/param/baseline.txt"
        scale = np.loadtxt(fileS, delimiter=',')
        baseline = np.loadtxt(fileB, delimiter=',')
        CCW_baseline = baseline - 0.01  # reduce to turn less

        turn__vel = 20
        delta_theta = compute_delta_theta([0.4, 0.4], robot_pose)
        CCW_turn_time = float( (abs(delta_theta)*CCW_baseline) / (2*turn__vel*scale) )
        take_pic_update_slam(operate, command=[0,1], turning_tick=turn__vel, time=CCW_turn_time)
        ##### CCW calibration #####



def look_for_ARUCO(operate):
    fileS = "calibration/param/scale.txt"
    scale = np.loadtxt(fileS, delimiter=',')
    fileB = "calibration/param/baseline.txt"
    baseline = np.loadtxt(fileB, delimiter=',')

    CCW_baseline = baseline - 0.01
    # reduce to turn less

    # print(f'Before turn: {robot_pose}')
    turn__vel = 20
    delta_theta = float(np.pi / 4)
    # margin = 2 * np.pi / 180
    # cnt = 0

    CCW_turn_time = float( (abs(delta_theta)*CCW_baseline) / (2*turn__vel*scale) )
    take_pic_update_slam(operate, command=[0,1], turning_tick=turn__vel, time=CCW_turn_time)
    time.sleep(0.5)








if __name__ == "__main__":
    parser = argparse.ArgumentParser("Fruit searching")
    parser.add_argument("--map", type=str, default='tuesday_3_fruit.txt')
    parser.add_argument("--ip", metavar='', type=str, default='localhost')
    parser.add_argument("--port", metavar='', type=int, default=8000)
    parser.add_argument("--calib_dir", type=str, default="calibration/param/")
    parser.add_argument("--save_data", action='store_true')
    parser.add_argument("--play_data", action='store_true')
    parser.add_argument("--ckpt", default='network/scripts/model/YOLOYbest.pt')
    args, _ = parser.parse_known_args()

    # GUI
    pygame.font.init()
    width, height = 902, 660
    canvas = pygame.display.set_mode((width, height))
    pygame.display.set_caption('ECE4078 2022 Lab')
    pygame.display.set_icon(pygame.image.load('pics/8bit/pibot5.png'))
    canvas.fill((0, 0, 0))
    splash = pygame.image.load('pics/loading.png')
    pibot_animate = [pygame.image.load('pics/8bit/pibot1.png'),
                     pygame.image.load('pics/8bit/pibot2.png'),
                     pygame.image.load('pics/8bit/pibot3.png'),
                    pygame.image.load('pics/8bit/pibot4.png'),
                     pygame.image.load('pics/8bit/pibot5.png')]
    pygame.display.update()

    # debugging parameter
    debug_djikstra = False

    ppi = Alphabot(args.ip, args.port)
    operate = Operate(args, ppi)

    # read in the true map
    fruits_list, fruits_true_pos, aruco_true_pos = read_true_map(args.map)
    search_list = read_search_list()
    print_target_fruits_pos(search_list, fruits_list, fruits_true_pos)

    waypoint = [0.0,0.0]
    robot_pose = [0.0,0.0,0.0]

   

    # run SLAM
    n_observed_markers = len(operate.ekf.taglist)
    if n_observed_markers == 0:
        if not operate.ekf_on:
            print('SLAM is running')
            operate.ekf_on = True
        else:
            print('> 2 landmarks is required for pausing - 1st')
    elif n_observed_markers < 3:
        print('> 2 landmarks is required for pausing')
    else:
        if not operate.ekf_on:
            operate.request_recover_robot = True
        operate.ekf_on = not operate.ekf_on
        if operate.ekf_on:
            print('SLAM is running')
        else:
            print('SLAM is paused')

    if not debug_djikstra:
        take_pic_update_slam(operate)    

    # Initialise SLAM states with marker position from true map
    LMS = []
    for idx, marker in enumerate(aruco_true_pos):
        LMS.append(measure.Marker(np.expand_dims(aruco_true_pos[idx], -1), idx+1))
    operate.ekf.add_landmarks(LMS)



    # LOOP over each target fruit
    for fruit in search_list:
        for i in range(3):
            if fruit == fruits_list[i]:
                target_fruit_idx = i
        
        ##### FORWARD calibration #####
        # calibrate('forward')

        ##### CW calibration #####
        # calibrate('CW')

        ##### CCW calibration #####
        # calibrate('CCW')

        

        # create occupancy map
        obs_x, obs_y = create_occupancy_map(fruits_true_pos, aruco_true_pos, target_fruit_idx)

        # Dijkstra path planning algorithm
        waypts_x, waypts_y = run_Djikstra(operate, fruits_true_pos, target_fruit_idx, obs_x, obs_y, plot_path=debug_djikstra)
        
        # LOOP over each waypoints
        x, y = 0.0, 0.0
        for i in range(1, len(waypts_x)-1): # loop the navigation waypoint
            x = waypts_x[-i-1]
            y = waypts_y[-i-1]

            # estimate the robot's pose
            robot_pose = get_robot_pose(operate)
            
            # robot drives to the waypoint
            waypoint = [x,y]
            drive_to_point(waypoint, robot_pose, operate)
            take_pic_update_slam(operate)
            robot_pose = get_robot_pose(operate)
            print("Finished driving to waypoint: {}; New robot pose: {}".format(waypoint,robot_pose))

            if i == len(waypts_x) - 2:  # reached target fruit
                for _ in range(8):
                    look_for_ARUCO(operate)
                time.sleep(1)

        # exit
        ppi.set_velocity([0, 0])
        # uInput = input("Move to the next target fruit? [Y/N]")
        # if uInput == 'N':
        #     break  
        