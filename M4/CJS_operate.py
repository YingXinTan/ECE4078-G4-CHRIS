import numpy as np
import cv2 
import os, sys
import time
from time import sleep

# import utility functions
sys.path.insert(0, "{}/utility".format(os.getcwd()))
from util.pibot import Alphabot # access the robot
import util.DatasetHandler as dh # save/load functions
import util.measure as measure # measurements
import pygame # python package for GUI
import shutil # python package for file operations
from dijkstra import Dijkstra
import matplotlib.pyplot as plt

# import SLAM components you developed in M2
sys.path.insert(0, "{}/slam".format(os.getcwd()))
from slam.ekf import EKF
from slam.robot import Robot
import slam.aruco_detector as aruco

# import CV components
sys.path.insert(0,"{}/network/".format(os.getcwd()))
sys.path.insert(0,"{}/network/scripts".format(os.getcwd()))
from network.scripts.detector import Detector

class Operate:
    def __init__(self, args):
        # self.args = args
        # self.canvas = canvas

        self.folder = 'pibot_dataset/'
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)
        else:
            shutil.rmtree(self.folder)
            os.makedirs(self.folder)
        
        # # initialise data parameters
        # if args.play_data:
        #     self.pibot = dh.DatasetPlayer("record")
        # else:
        #     self.pibot = Alphabot(args.ip, args.port)
        self.pibot = Alphabot(args.ip, args.port)

        # initialise SLAM parameters
        self.ekf = self.init_ekf(args.calib_dir, args.ip)
        self.aruco_det = aruco.aruco_detector(
            self.ekf.robot, marker_length = 0.06) # size of the ARUCO markers

        if args.save_data:
            self.data = dh.DatasetWriter('record')
        else:
            self.data = None
        self.output = dh.OutputWriter('lab_output')
        self.command = {'motion':[0, 0], 
                        'inference': False,
                        'output': False,
                        'save_inference': False,
                        'save_image': False}
        self.quit = False
        self.pred_fname = ''
        self.request_recover_robot = False
        self.file_output = None
        self.ekf_on = False
        self.double_reset_comfirm = 0
        self.image_id = 0
        self.notification = 'Press ENTER to start SLAM'
        # a 5min timer
        # self.count_down = 300
        # self.start_time = time.time()
        self.control_clock = time.time()
        # initialise images
        self.img = np.zeros([240,320,3], dtype=np.uint8)
        self.aruco_img = np.zeros([240,320,3], dtype=np.uint8)
        self.detector_output = np.zeros([240,320], dtype=np.uint8)
        if args.ckpt == "":
            self.detector = None
            self.network_vis = cv2.imread('pics/8bit/detector_splash.png')
        else:
            # self.detector = Detector(args.ckpt, use_gpu=False)
            self.detector = None
            self.network_vis = np.ones((240, 320,3))* 100
        self.bg = pygame.image.load('pics/gui_mask.jpg')

        self.fileS = "calibration/param/scale.txt"
        self.scale = np.loadtxt(self.fileS, delimiter=',')
        self.fileB = "calibration/param/baseline.txt"
        self.baseline = np.loadtxt(self.fileB, delimiter=',')      

    # wheel control
    def control(self, tick=0, turning_tick=0):       
        # if self.args.play_data:
        #     lv, rv = self.pibot.set_velocity()            
        # else:
        #     lv, rv = self.pibot.set_velocity(
        #         self.command['motion'])
        lv, rv = self.pibot.set_velocity(self.command['motion'], tick=tick, turning_tick=turning_tick, time=0)
        if not self.data is None:
            self.data.write_keyboard(lv, rv)
        dt = time.time() - self.control_clock
        drive_meas = measure.Drive(lv, rv, dt)
        self.control_clock = time.time()
        return drive_meas

    # camera control
    def take_pic(self):
        self.img = self.pibot.get_image()
        if not self.data is None:
            self.data.write_image(self.img)

    # SLAM with ARUCO markers       
    def update_slam(self, drive_meas):
        lms, self.aruco_img = self.aruco_det.detect_marker_positions(self.img)
        if self.request_recover_robot:
            is_success = self.ekf.recover_from_pause(lms)
            if is_success:
                self.notification = 'Robot pose is successfuly recovered'
                self.ekf_on = True
            else:
                self.notification = 'Recover failed, need >2 landmarks!'
                self.ekf_on = False
            self.request_recover_robot = False
        elif self.ekf_on: # and not self.debug_flag:
            self.ekf.predict(drive_meas)
            self.ekf.add_landmarks(lms)
            self.ekf.update(lms)

    # # using computer vision to detect targets
    # def detect_target(self):
    #     if self.command['inference'] and self.detector is not None:
    #         self.detector_output, self.network_vis = self.detector.detect_single_image_yolo(self.img)
    #         self.command['inference'] = False
    #         self.file_output = (self.detector_output, self.ekf)
    #         self.notification = f'{len(np.unique(self.detector_output))-1} target type(s) detected'

    # # save raw images taken by the camera
    # def save_image(self):
    #     f_ = os.path.join(self.folder, f'img_{self.image_id}.png')
    #     if self.command['save_image']:
    #         image = self.pibot.get_image()
    #         image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    #         cv2.imwrite(f_, image)
    #         self.image_id += 1
    #         self.command['save_image'] = False
    #         self.notification = f'{f_} is saved'

    # wheel and camera calibration for SLAM
    def init_ekf(self, datadir, ip):
        fileK = "{}intrinsic.txt".format(datadir)
        camera_matrix = np.loadtxt(fileK, delimiter=',')
        fileD = "{}distCoeffs.txt".format(datadir)
        dist_coeffs = np.loadtxt(fileD, delimiter=',')
        fileS = "{}scale.txt".format(datadir)
        scale = np.loadtxt(fileS, delimiter=',')
        if ip == 'localhost':
            scale /= 2
        fileB = "{}baseline.txt".format(datadir)  
        baseline = np.loadtxt(fileB, delimiter=',')
        robot = Robot(baseline, scale, camera_matrix, dist_coeffs)
        return EKF(robot)

    # # save SLAM map
    # def record_data(self):
    #     if self.command['output']:
    #         self.output.write_map(self.ekf)
    #         self.notification = 'Map is saved'
    #         self.command['output'] = False
    #     # save inference with the matching robot pose and detector labels
    #     if self.command['save_inference']:
    #         if self.file_output is not None:
    #             #image = cv2.cvtColor(self.file_output[0], cv2.COLOR_RGB2BGR)
    #             self.pred_fname = self.output.write_image(self.file_output[0],
    #                                                     self.file_output[1])
    #             self.notification = f'Prediction is saved to {operate.pred_fname}'
    #         else:
    #             self.notification = f'No prediction in buffer, save ignored'
    #         self.command['save_inference'] = False

    # paint the GUI            
    def draw(self, canvas):
        TEXT_FONT = pygame.font.Font('pics/8-BitMadness.ttf', 40)
        canvas.blit(self.bg, (0, 0))
        text_colour = (220, 220, 220)
        v_pad = 40
        h_pad = 20

        # paint SLAM outputs
        ekf_view = self.ekf.draw_slam_state(res=(320, 480+v_pad),
            not_pause = self.ekf_on)
        canvas.blit(ekf_view, (2*h_pad+320, v_pad))
        robot_view = cv2.resize(self.aruco_img, (320, 240))
        self.draw_pygame_window(canvas, robot_view, 
                                position=(h_pad, v_pad)
                                )

        # for target detector (M3)
        detector_view = cv2.resize(self.network_vis,
                                   (320, 240), cv2.INTER_NEAREST)
        # print(detector_view.shape)
        self.draw_pygame_window(canvas, detector_view, 
                                position=(h_pad, 240+2*v_pad)
                                )

        # canvas.blit(self.gui_mask, (0, 0))
        self.put_caption(canvas, caption='SLAM', position=(2*h_pad+320, v_pad))
        self.put_caption(canvas, caption='Detector',
                         position=(h_pad, 240+2*v_pad))
        self.put_caption(canvas, caption='PiBot Cam', position=(h_pad, v_pad))

        notification = TEXT_FONT.render(self.notification,
                                          False, text_colour)
        canvas.blit(notification, (h_pad+10, 596))

        # time_remain = self.count_down - time.time() + self.start_time
        # if time_remain > 0:
        #     time_remain = f'Count Down: {time_remain:03.0f}s'
        # elif int(time_remain)%2 == 0:
        #     time_remain = "Time Is Up !!!"
        # else:
        #     time_remain = ""
        # count_down_surface = TEXT_FONT.render(time_remain, False, (50, 50, 50))
        # canvas.blit(count_down_surface, (2*h_pad+320+5, 530))
        return canvas

    @staticmethod
    def draw_pygame_window(canvas, cv2_img, position):
        cv2_img = np.rot90(cv2_img)
        view = pygame.surfarray.make_surface(cv2_img)
        view = pygame.transform.flip(view, True, False)
        canvas.blit(view, position)
    
    @staticmethod
    def put_caption(canvas, caption, position, text_colour=(200, 200, 200)):
        TITLE_FONT = pygame.font.Font('pics/8-BitMadness.ttf', 35)
        caption_surface = TITLE_FONT.render(caption,
                                          False, text_colour)
        canvas.blit(caption_surface, (position[0], position[1]-25))
    
    def get_robot_pose(self):
        return self.ekf.get_state_vector()[0:3,:]

    # obstacle map
    def create_occupancy_map(self, fruits_true_pos, aruco_true_pos, target_fruit_idx):
        ox, oy = [], [] # obstacle location

        # define obstacle fruit's location
        for i in range(5):
            if i == target_fruit_idx: # do not include the current target fruit as obstacle
                continue
            ox.append(fruits_true_pos[i][0])
            oy.append(fruits_true_pos[i][1])
        
        # define aruco obstacle location
        for i in range(10):
            ox.append(aruco_true_pos[i][0])
            oy.append(aruco_true_pos[i][1])
        return ox, oy

    # path planning
    def run_Djikstra(self, fruits_true_pos, target_fruit_idx, obs_x, obs_y, plot_path=False):
        # compute robot radius from baseline
        # CHANGE HERE !!!!!!!!!!!!
        # robot_radius = float( np.loadtxt("calibration/param/baseline.txt", delimiter=',') / 2 ) * 3.5
        # robot_radius = float( np.loadtxt("calibration/param/baseline.txt", delimiter=',') / 2 ) * 4.0
        robot_radius = float( np.loadtxt("calibration/param/baseline.txt", delimiter=',') / 2 ) * 3.8

        # occupancy grid resolution [m]
        grid_resolution = 0.2

        # current robot pose as starting position 
        robot_pose = self.get_robot_pose()
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
            # plt.show()
        return waypts_x, waypts_y
    
    # compute delta_theta
    def compute_delta_theta(self, waypoint):
        robot_pose = self.get_robot_pose()
        theta_goal = np.arctan2(waypoint[1]-robot_pose[1], waypoint[0]-robot_pose[0])
        delta_theta = theta_goal - robot_pose[2]    
        if delta_theta < -np.pi:
            delta_theta += 2*np.pi
        elif delta_theta > np.pi:
            delta_theta -= 2*np.pi
        return delta_theta[0]
    
    # def drive_to_point(self, waypoint):
    #     robot_pose = self.get_robot_pose()
    #     # print(f'Before turn: {get_robot_pose(operate)}')

    #     turning(waypoint, robot_pose, operate)

    #     robot_pose = get_robot_pose(operate)
    #     # print(f'\nAfter turn: {robot_pose}')

    #     forward(waypoint, robot_pose, operate)

    #     # print(f'\nAfter forward: {get_robot_pose(operate)}')
    
    def update_gui(self, canvas):
        self.draw(canvas)
        pygame.display.update()

    
    def drive_take_pic_update_slam(self, command=[0,0], tick=0.0, turning_tick=0.0, time=0.0, delta_theta=None):
        # forward
        if command[0] == 1:
            calib_time = -0.1    # LOW battery 0.3?
            # calib_fraction = 0.8
            mode=1
        # CCW turn
        elif command[1] == 1:
            calib_time_1 = 0.0025   # 45
            # calib_time_1 = 0.005   # 45
            calib_time_2 = -0.09    # 135

            m = float( (calib_time_1 - calib_time_2) / (-1.570796) )
            c = float(calib_time_1 - m * 45 * np.pi / 180.0)
            if delta_theta is not None:
                # calib_time = 0.0
                calib_time = m * abs(delta_theta) + c
            # calib_time = calib_time_1
            # calib_time = calib_time_2
            # print(calib_time)
            # calib_fraction = 0.9
            mode=2
        # CW turn
        elif command[1] == -1:
            calib_time_1 = 0.02   # 45
            calib_time_2 = -0.008    # 135

            m = float( (calib_time_1 - calib_time_2) / (-1.570796) )
            c = calib_time_1 - m * 45 * np.pi / 180.0
            if delta_theta is not None:
                # calib_time = 0.0
                calib_time = m * abs(delta_theta) + c
            # print(calib_time)
            # calib_time = calib_time_1
            # calib_time = calib_time_2
            # calib_fraction = 0.9
            mode=3
        
        total_time = time + calib_time
        # print('out', total_time)

        # drive robot        
        lv, rv = self.pibot.set_velocity(command, tick=tick, turning_tick=turning_tick, time=total_time, calib_for_straight=True, mode=mode)
        # lv, rv = self.pibot.set_velocity(command, tick=tick, turning_tick=turning_tick, time=calib_fraction*total_time, calib_for_straight=True, mode=mode)

        # drive robot        
        # _, _ = self.pibot.set_velocity(command, tick=tick, turning_tick=turning_tick, time=(1.0-calib_fraction)*total_time, calib_for_straight=False, mode=mode)

        # sleep before taking pic
        sleep(0.5)
        self.take_pic()

        # update slam with Drive measure
        drive_meas = measure.Drive(lv, rv, time)
        self.update_slam(drive_meas)
        # operate.draw(canvas)
        # pygame.display.update()
        # operate.notification = f'{operate.ekf.get_state_vector()[0:3, :]}'
    
    def go_to_waypoint(self, waypoint):
        # to move more, reduce scale
        scale = self.scale
        robot_pose = self.get_robot_pose()

        wheel_vel = 20
        delta_d = np.sqrt((waypoint[0]-robot_pose[0])**2 + (waypoint[1]-robot_pose[1])**2)

        drive_time = float( delta_d / (wheel_vel*scale) )
        self.drive_take_pic_update_slam(command=[1,0], tick=wheel_vel, time=drive_time)
        # time.sleep(1)
    
    # call this function to get the value for `forward: calib_time`
    def forward_calibrator(self, fwd_distance=0.2, fwd_count=5):
        for _ in range(fwd_count):
            waypoint = [fwd_distance, 0.0]
            self.go_to_waypoint(waypoint)
            self.ekf.reset()
            sleep(0.5)

    def turn_to_waypoint(self, waypoint):
        # to move more, reduce scale
        scale = self.scale
        baseline = self.baseline
        
        # CW_baseline = baseline
        # CCW_baseline = baseline
        # reduce to turn less

        turn_vel = 20                
        delta_theta = self.compute_delta_theta(waypoint)

        turn_time = float( (abs(delta_theta)*baseline) / (2*turn_vel*scale) )
        if delta_theta > 0:
            self.drive_take_pic_update_slam(command=[0,1], turning_tick=turn_vel, time=turn_time, delta_theta=delta_theta)
        elif delta_theta < 0:
            self.drive_take_pic_update_slam(command=[0,-1], turning_tick=turn_vel, time=turn_time, delta_theta=delta_theta)

        # CW_turn_time = float( (abs(delta_theta)*CW_baseline) / (2*turn_vel*scale) )
        # print("Turning for {:.2f} seconds".format(turn_time))

        # if delta_theta > 0:
            
        # elif delta_theta < 0:
        #     drive_take_pic_update_slam(operate, command=[0,-1], turning_tick=turn__vel, time=CW_turn_time)    
        # # time.sleep(1)

        # # delta_d = np.sqrt((waypoint[0]-robot_pose[0])**2 + (waypoint[1]-robot_pose[1])**2)
        # margin = 0.01
        # cnt = 0

        # drive_time = float( delta_d / (wheel_vel*scale) )
        # self.drive_take_pic_update_slam(command=[1,0], tick=wheel_vel, time=drive_time)
        # time.sleep(1)
    
    # call this function to get the value for `calib_time`
    def turning_calibrator(self, turning_angle=45, turning_count=5, mode='CCW'):
        for _ in range(turning_count):
            if turning_angle == 45:
                x, y = 0.4, 0.4
            elif turning_angle == 90:
                x, y = 0.0, 0.4
            elif turning_angle == 135:
                x, y = -0.4, 0.4
            elif turning_angle == 180:
                x, y = -0.4, 1e-6
            
            if mode == 'CW':
                y = -1 * y
            waypoint = [x, y]
            self.turn_to_waypoint(waypoint)
            self.ekf.reset()
            sleep(0.5)

            
        

            
        

    # # P controller for turning
    # def turning_controller(self, delta_theta):
    #     min_vel = 10.0
    #     K_pw = 0.25

    #     turn_vel = float(min_vel + K_pw * delta_theta)
    #     # turn_vel = 15.0
    #     # if abs(delta_theta) <= abs(init_DD) 0.349066:
    #     # if abs(delta_theta) <= 0.2*abs(init_DD):
    #     #     turn_vel = 10.0
    #     # turn_vel = 10.0
    #     # print(K_pw)
    #     # print(delta_theta)
    #     # print(turn_vel)
    #     # if turn_vel > max_vel:
    #         # turn_vel = max_vel
    #     # if turn_vel < min_vel:
    #     #     turn_vel = min_vel
        
    #     if delta_theta > 0:
    #         self.command['motion'] = [0,1]
    #     elif delta_theta < 0:
    #         self.command['motion'] = [0,-1]
    #     return turn_vel
    
    # # compute delta_theta
    # def compute_delta_d(self, waypoint):
    #     robot_pose = self.get_robot_pose()
    #     return np.sqrt((waypoint[0]-robot_pose[0])**2 + (waypoint[1]-robot_pose[1])**2)
    
    # # P controller for forward
    # def forward_controller(self, delta_d, prev_delta_d):
    #     min_vel = 10.0
    #     K_pw = 0.25

    #     forward_vel = float(min_vel + K_pw * delta_d)
    #     # print('4ward vel', forward_vel)
    #     # forward_vel = 15.0
    #     # if abs(delta_theta) <= abs(init_DD) 0.349066:
    #     # if abs(delta_theta) <= 0.2*abs(init_DD):
    #     #     turn_vel = 10.0
    #     # turn_vel = 10.0
    #     # print(K_pw)
    #     # print(delta_theta)
    #     # print(turn_vel)
    #     # if turn_vel > max_vel:
    #         # turn_vel = max_vel
    #     # if turn_vel < min_vel:
    #     #     turn_vel = min_vel
        
    #     if delta_d > prev_delta_d:
    #         self.command['motion'] = [0,0]
    #     else:
    #         self.command['motion'] = [1,0]
    #     # self.command['motion'] = [1,0]
    #     return forward_vel
    
    def look_for_ARUCO(self):
        # to move more, reduce scale
        scale = self.scale
        baseline = self.baseline
        
        # CW_baseline = baseline
        # CCW_baseline = baseline
        # reduce to turn less

        turn_vel = 20                
        # delta_theta = np.pi/4     # turn 45 degrees
        delta_theta = np.pi/6       # turn 30 degrees
        #  self.compute_delta_theta(waypoint)

        turn_time = float( (abs(delta_theta)*baseline) / (2*turn_vel*scale) )
        self.drive_take_pic_update_slam(command=[0,1], turning_tick=turn_vel, time=turn_time, delta_theta=delta_theta)



    # def forward(waypoint, robot_pose, operate):
    #     fileS = "calibration/param/scale.txt"
    #     # to move more, reduce scale
    #     scale = np.loadtxt(fileS, delimiter=',')

    #     wheel_vel = 22
    #     # delta_d = np.sqrt((waypoint[0]-robot_pose[0])**2 + (waypoint[1]-robot_pose[1])**2) + 0.04
    #     delta_d = np.sqrt((waypoint[0]-robot_pose[0])**2 + (waypoint[1]-robot_pose[1])**2)
    #     margin = 0.01
    #     # cnt = 0

    #     drive_time = float( delta_d / (wheel_vel*scale) )
    #     drive_take_pic_update_slam(operate, command=[1,0], tick=wheel_vel, time=drive_time)
    #     time.sleep(1)

    #     # while delta_d > margin:        
    #     #     drive_time = float( delta_d / (wheel_vel*scale) )
    #     #     drive_take_pic_update_slam(operate, command=[1,0], tick=wheel_vel, time=drive_time)
    #     #     time.sleep(1)

    #     #     robot_pose = get_robot_pose(operate)
    #     #     delta_d = np.sqrt((waypoint[0]-robot_pose[0])**2 + (waypoint[1]-robot_pose[1])**2)
    #     #     # cnt+=1
    #     # # print(f'\nRandike Forward Cnt: {cnt}')

    # # keyboard teleoperation        
    # def update_keyboard(self):
    #     for event in pygame.event.get():
    #         ########### replace with your M1 codes ###########
    #         lin_speed = 2
    #         ang_speed = 3
    #         # drive forward
    #         if event.type == pygame.KEYDOWN and event.key == pygame.K_UP:
    #             # pass # TODO: replace with your M1 code
    #             self.command['motion'] = [lin_speed, 0]
    #         # drive backward
    #         elif event.type == pygame.KEYDOWN and event.key == pygame.K_DOWN:
    #             # pass # TODO: replace with your M1 code
    #             self.command['motion'] = [-lin_speed, 0]
    #         # turn left
    #         elif event.type == pygame.KEYDOWN and event.key == pygame.K_LEFT:
    #             # pass # TODO: replace with your M1 code
    #             self.command['motion'] = [0, ang_speed]
    #         # drive right
    #         elif event.type == pygame.KEYDOWN and event.key == pygame.K_RIGHT:
    #             # pass # TODO: replace with your M1 code
    #             self.command['motion'] = [0, -ang_speed]
    #         ####################################################
    #         # stop
    #         elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
    #             self.command['motion'] = [0, 0]
    #         # save image
    #         elif event.type == pygame.KEYDOWN and event.key == pygame.K_i:
    #             self.command['save_image'] = True
    #         # save SLAM map
    #         elif event.type == pygame.KEYDOWN and event.key == pygame.K_s:
    #             self.command['output'] = True
    #         # reset SLAM map
    #         elif event.type == pygame.KEYDOWN and event.key == pygame.K_r:
    #             if self.double_reset_comfirm == 0:
    #                 self.notification = 'Press again to confirm CLEAR MAP'
    #                 self.double_reset_comfirm +=1
    #             elif self.double_reset_comfirm == 1:
    #                 self.notification = 'SLAM Map is cleared'
    #                 self.double_reset_comfirm = 0
    #                 self.ekf.reset()
    #         # run SLAM
    #         elif event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN:
    #             n_observed_markers = len(self.ekf.taglist)
    #             if n_observed_markers == 0:
    #                 if not self.ekf_on:
    #                     self.notification = 'SLAM is running'
    #                     self.ekf_on = True
    #                 else:
    #                     self.notification = '> 2 landmarks is required for pausing'
    #             elif n_observed_markers < 3:
    #                 self.notification = '> 2 landmarks is required for pausing'
    #             else:
    #                 if not self.ekf_on:
    #                     self.request_recover_robot = True
    #                 self.ekf_on = not self.ekf_on
    #                 if self.ekf_on:
    #                     self.notification = 'SLAM is running'
    #                 else:
    #                     self.notification = 'SLAM is paused'
    #         # run object detector
    #         elif event.type == pygame.KEYDOWN and event.key == pygame.K_p:
    #             self.command['inference'] = True
    #         # save object detection outputs
    #         elif event.type == pygame.KEYDOWN and event.key == pygame.K_n:
    #             self.command['save_inference'] = True
    #         # quit
    #         elif event.type == pygame.QUIT:
    #             self.quit = True
    #         elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
    #             self.quit = True
    #     if self.quit:
    #         pygame.quit()
    #         sys.exit()

        
# if __name__ == "__main__":
    # import argparse

    # parser = argparse.ArgumentParser()
    # parser.add_argument("--ip", metavar='', type=str, default='localhost')
    # parser.add_argument("--port", metavar='', type=int, default=8000)
    # parser.add_argument("--calib_dir", type=str, default="calibration/param/")
    # parser.add_argument("--save_data", action='store_true')
    # parser.add_argument("--play_data", action='store_true')
    # # parser.add_argument("--ckpt", default='network/scripts/model/model.best.pth')
    # parser.add_argument("--ckpt", default='network/scripts/model/YOLObest.pt')
    # args, _ = parser.parse_known_args()
    
    # pygame.font.init() 
    # TITLE_FONT = pygame.font.Font('pics/8-BitMadness.ttf', 35)
    # TEXT_FONT = pygame.font.Font('pics/8-BitMadness.ttf', 40)

    # # width, height = 700, 660
    # canvas = pygame.display.set_mode((700, 600))
    # pygame.display.set_caption('ECE4078 2021 Lab')
    # pygame.display.set_icon(pygame.image.load('pics/8bit/pibot5.png'))
    # canvas.fill((0, 0, 0))
    # splash = pygame.image.load('pics/loading.png')
    # pibot_animate = [pygame.image.load('pics/8bit/pibot1.png'),
    #             pygame.image.load('pics/8bit/pibot2.png'),
    #             pygame.image.load('pics/8bit/pibot3.png'),
    #             pygame.image.load('pics/8bit/pibot4.png'),
    #             pygame.image.load('pics/8bit/pibot5.png')]
    # pygame.display.update()

        # running = True
        # while running:
        #     for event in pygame.event.get():
        #         pass

    # start = False

    # counter = 40
    # while not start:
    #     for event in pygame.event.get():
    #         if event.type == pygame.KEYDOWN:
    #             start = True
    #     canvas.blit(splash, (0, 0))
    #     x_ = min(counter, 600)
    #     if x_ < 600:
    #         canvas.blit(pibot_animate[counter%10//2], (x_, 565))
    #         pygame.display.update()
    #         counter += 2

    # operate = Operate(args)

    # while True:
    #     operate.update_keyboard()
    #     operate.take_pic()
    #     drive_meas = operate.control()
    #     operate.update_slam(drive_meas)
    #     operate.record_data()
    #     operate.save_image()
    #     operate.detect_target()
    #     # visualise
    #     operate.draw(canvas)
    #     pygame.display.update()




