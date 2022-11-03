# M4 - Autonomous fruit searching

# basic python packages
import sys, os
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
    scale = np.loadtxt(fileS, delimiter=',') + 0.001

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


# # Waypoint navigation
# # the robot automatically drives to a given [x,y] coordinate
# # additional improvements:
# # you may use different motion model parameters for robot driving on its own or driving while pushing a fruit
# # try changing to a fully automatic delivery approach: develop a path-finding algorithm that produces the waypoints
# def drive_to_point(waypoint, robot_pose, operate):
#     # imports camera / wheel calibration parameters 
#     fileS = "calibration/param/scale.txt"
#     scale = np.loadtxt(fileS, delimiter=',')
#     fileB = "calibration/param/baseline.txt"
#     baseline = np.loadtxt(fileB, delimiter=',')
    
#     ####################################################
#     # TODO: replace with your codes to make the robot drive to the waypoint
#     # One simple strategy is to first turn on the spot facing the waypoint,
#     # then drive straight to the way point

#     wheel_vel = 20 # tick to move the robot
#     # lin_spd = 2 * wheel_vel
#     # ang_spd = 3
    
#     # turn towards the waypoint
#     theta_goal = np.arctan2(waypoint[1]-robot_pose[1], waypoint[0]-robot_pose[0])
#     delta_theta = theta_goal - robot_pose[2]
#     turn_time = float( (abs(delta_theta)*baseline) / (2*wheel_vel*scale) )
#     print("Turning for {:.2f} seconds".format(turn_time))
#     if delta_theta > 0:
#         take_pic_update_slam(operate, command=[0,1], turning_tick=wheel_vel, time=turn_time)
#     elif delta_theta < 0:
#         take_pic_update_slam(operate, command=[0,-1], turning_tick=wheel_vel, time=turn_time)

#     # after turning, drive straight to the waypoint
#     delta_d = np.sqrt((waypoint[0]-robot_pose[0])**2 + (waypoint[1]-robot_pose[1])**2)
#     drive_time = float( delta_d / (wheel_vel*scale) )
#     print("Driving for {:.2f} seconds".format(drive_time))
#     take_pic_update_slam(operate, command=[1,0], tick=wheel_vel, time=drive_time)
#     ####################################################
#     print("Arrived at [{}, {}]".format(waypoint[0], waypoint[1]))


# def get_robot_pose(operate):
#     ####################################################
#     # TODO: replace with your codes to estimate the pose of the robot
#     # We STRONGLY RECOMMEND you to use your SLAM code from M2 here

#     # update the robot pose [x,y,theta]
#     robot_pose = operate.ekf.get_state_vector()[0:3,:]
#     ####################################################
#     return robot_pose


# def take_pic_update_slam(operate, command=[0,0], tick=0.0, turning_tick=0.0, time=0.0):
#     operate.take_pic()
#     lv, rv = operate.pibot.set_velocity(command, tick=tick, turning_tick=turning_tick, time=time)
#     drive_meas = measure.Drive(lv, rv, time)
#     operate.update_slam(drive_meas)



# main loop
if __name__ == "__main__":
    parser = argparse.ArgumentParser("Fruit searching")
    parser.add_argument("--map", type=str, default='M4_true_map.txt')
    parser.add_argument("--ip", metavar='', type=str, default='localhost')
    parser.add_argument("--port", metavar='', type=int, default=8000)

    # --------- copied from operate.py ---------
    # parser.add_argument("--ip", metavar='', type=str, default='localhost')
    # parser.add_argument("--port", metavar='', type=int, default=8000)
    parser.add_argument("--calib_dir", type=str, default="calibration/param/")
    parser.add_argument("--save_data", action='store_true')
    # parser.add_argument("--play_data", action='store_true')
    # parser.add_argument("--ckpt", default='network/scripts/model/model.best.pth')
    # --------- copied from operate.py ---------

    args, _ = parser.parse_known_args()
    ppi = Alphabot(args.ip,args.port)

    # ----------------------------- Newly added Initialisation -----------------------------
    operate = Operate(args, ppi)
    # ----------------------------- Newly added Initialisation -----------------------------

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

    take_pic_update_slam(operate)

    # The following code is only a skeleton code the semi-auto fruit searching task
    while True:

        # enter the waypoints
        # instead of manually enter waypoints in command line, you can get coordinates by clicking on a map (GUI input), see camera_calibration.py
        x,y = 0.0,0.0
        x = input("X coordinate of the waypoint: ")
        try:
            x = float(x)
        except ValueError:
            print("Please enter a number.")
            continue
        y = input("Y coordinate of the waypoint: ")
        try:
            y = float(y)
        except ValueError:
            print("Please enter a number.")
            continue

        
        # estimate the robot's pose
        robot_pose = get_robot_pose(operate)

        
        # robot drives to the waypoint
        waypoint = [x,y]
        drive_to_point(waypoint, robot_pose, operate)
        robot_pose = get_robot_pose(operate)
        print("Finished driving to waypoint: {}; New robot pose: {}".format(waypoint,robot_pose))


        # exit
        ppi.set_velocity([0, 0])
        uInput = input("Add a new waypoint? [Y/N]")
        if uInput == 'N':
            break