# teleoperate the robot, perform SLAM and object detection

# basic python packages
from ast import arg
import ast
import json
import numpy as np
import cv2 
import os, sys
import time

# import utility functions
sys.path.insert(0, "{}/utility".format(os.getcwd()))
from util.pibot import Alphabot # access the robot
import util.DatasetHandler as dh # save/load functions
import util.measure as measure # measurements
import pygame # python package for GUI
import shutil # python package for file operations

# import SLAM components you developed in M2
sys.path.insert(0, "{}/slam".format(os.getcwd()))
from slam.ekf import EKF
from slam.robot import Robot
import slam.aruco_detector as aruco

# import CV components
sys.path.insert(0,"{}/network/".format(os.getcwd()))
sys.path.insert(0,"{}/network/scripts".format(os.getcwd()))
from network.scripts.detector import Detector
# from TargetPoseEst import get_fruit_pose_RF

os.environ['KMP_DUPLICATE_LIB_OK']='True' # This makes the code unstable, don't use if don't need. 

class Operate:
    def __init__(self, args):
        self.folder = 'pibot_dataset/'
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)
        else:
            shutil.rmtree(self.folder)
            os.makedirs(self.folder)
        
        # initialise data parameters
        if args.play_data:
            self.pibot = dh.DatasetPlayer("record")
        else:
            self.pibot = Alphabot(args.ip, args.port)

        # initialise SLAM parameters
        self.ekf = self.init_ekf(args.calib_dir, args.ip)
        self.aruco_det = aruco.aruco_detector(
            self.ekf.robot, marker_length = 0.06) # size of the ARUCO markers

        if args.save_data:
            self.data = dh.DatasetWriter('record')
        else:
            self.data = None
        self.output = dh.OutputWriter(folder_name='lab_output',mode=1) # mode 1: for ARUCO SLAM.txt generation      mode 2: for FRUITS 
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
        self.count_down = 300
        self.start_time = time.time()
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

        self.low_speed_mode = False

    # wheel control
    def control(self):       
        if args.play_data:
            lv, rv = self.pibot.set_velocity()          
        else:
            # lv, rv = self.pibot.set_velocity(
                # self.command['motion'])
            
            # calibrated control
            mode = 0
            calib_for_straight = False
            
            if self.command['motion'][0] == 0 and self.command['motion'][1] == 0:
                calib_for_straight = False
            if self.command['motion'][0] == 1:
                mode = 1
                calib_for_straight=True
            elif self.command['motion'][-1] == 1:
                mode = 2   # CCW turning
                calib_for_straight=True
            elif self.command['motion'][-1] == -1:
                mode = 3   # CW turning
                calib_for_straight=True
            lv, rv = self.pibot.set_velocity(self.command['motion'], calib_for_straight=calib_for_straight, mode=mode, low_speed_mode = self.low_speed_mode)

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
                pass
            self.request_recover_robot = False
        elif self.ekf_on: # and not self.debug_flag:
            self.ekf.predict(drive_meas)
            self.ekf.add_landmarks(lms)
            self.ekf.update(lms)

    # using computer vision to detect targets
    def detect_target(self):
        if self.command['inference'] and self.detector is not None:
            self.detector_output, self.network_vis = self.detector.detect_single_image_yolo(self.img)
            self.command['inference'] = False
            self.file_output = (self.detector_output, self.ekf)
            self.notification = f'{len(np.unique(self.detector_output))-1} target type(s) detected'

    # save raw images taken by the camera
    def save_image(self):
        f_ = os.path.join(self.folder, f'img_{self.image_id}.png')
        if self.command['save_image']:
            image = self.pibot.get_image()
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(f_, image)
            self.image_id += 1
            self.command['save_image'] = False
            self.notification = f'{f_} is saved'

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

    # save SLAM map
    def record_data(self):
        if self.command['output']:
            self.output.write_map(self.ekf)

            # if args.mode == 1:  # first stage: SLAM
                # self.gen_est_map("lab_output/slam.txt", "aruco_map.txt")
                
            self.notification = 'Map is saved'
            self.command['output'] = False
        # save inference with the matching robot pose and detector labels
        if self.command['save_inference']:
            if self.file_output is not None:
                #image = cv2.cvtColor(self.file_output[0], cv2.COLOR_RGB2BGR)
                self.pred_fname = self.output.write_image(self.file_output[0],
                                                        self.file_output[1])
                self.notification = f'Prediction is saved to {operate.pred_fname}'
            else:
                self.notification = f'No prediction in buffer, save ignored'
            self.command['save_inference'] = False

    # paint the GUI            
    def draw(self, canvas):
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

        notifiation = TEXT_FONT.render(self.notification,
                                          False, text_colour)
        canvas.blit(notifiation, (h_pad+10, 596))

        time_remain = self.count_down - time.time() + self.start_time
        if time_remain > 0:
            time_remain = f'Count Down: {time_remain:03.0f}s'
        elif int(time_remain)%2 == 0:
            time_remain = "Time Is Up !!!"
        else:
            time_remain = ""
        count_down_surface = TEXT_FONT.render(time_remain, False, (50, 50, 50))
        canvas.blit(count_down_surface, (2*h_pad+320+5, 530))
        return canvas

    @staticmethod
    def draw_pygame_window(canvas, cv2_img, position):
        cv2_img = np.rot90(cv2_img)
        view = pygame.surfarray.make_surface(cv2_img)
        view = pygame.transform.flip(view, True, False)
        canvas.blit(view, position)
    
    @staticmethod
    def put_caption(canvas, caption, position, text_colour=(200, 200, 200)):
        caption_surface = TITLE_FONT.render(caption,
                                          False, text_colour)
        canvas.blit(caption_surface, (position[0], position[1]-25))

    # keyboard teleoperation        
    def update_keyboard(self):
        for event in pygame.event.get():
            ########### replace with your M1 codes ###########
            # lin_speed = 3
            # ang_speed = 2
            lin_speed = 1
            ang_speed = 1
            # drive forward
            if event.type == pygame.KEYDOWN and event.key == pygame.K_UP:
                # pass # TODO: replace with your M1 code
                self.command['motion'] = [lin_speed, 0]
            # drive backward
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_DOWN:
                # pass # TODO: replace with your M1 code
                self.command['motion'] = [-lin_speed, 0]
            # turn left
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_LEFT:
                # pass # TODO: replace with your M1 code
                self.command['motion'] = [0, ang_speed]
            # drive right
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_RIGHT:
                # pass # TODO: replace with your M1 code
                self.command['motion'] = [0, -ang_speed]
            
            # elif event.type == pygame.KEYUP and event.key == pygame.K_UP:
            #     # pass # TODO: replace with your M1 code
            #     self.command['motion'] = [0, 0]
            # # drive backward
            # elif event.type == pygame.KEYUP and event.key == pygame.K_DOWN:
            #     # pass # TODO: replace with your M1 code
            #     self.command['motion'] = [0, 0]
            # # turn left
            # elif event.type == pygame.KEYUP and event.key == pygame.K_LEFT:
            #     # pass # TODO: replace with your M1 code
            #     self.command['motion'] = [0, 0]
            # # drive right
            # elif event.type == pygame.KEYUP and event.key == pygame.K_RIGHT:
            #     # pass # TODO: replace with your M1 code
            #     self.command['motion'] = [0, 0]
            ####################################################
            # stop
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                self.command['motion'] = [0, 0]
            # save image
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_i:
                self.command['save_image'] = True
            # save SLAM map
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_s:
                self.command['output'] = True
            # reset SLAM map
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                if self.double_reset_comfirm == 0:
                    self.notification = 'Press again to confirm CLEAR MAP'
                    self.double_reset_comfirm +=1
                elif self.double_reset_comfirm == 1:
                    self.notification = 'SLAM Map is cleared'
                    self.double_reset_comfirm = 0
                    self.ekf.reset()
            # run SLAM
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN:
                n_observed_markers = len(self.ekf.taglist)
                print(n_observed_markers)
                if n_observed_markers == 0:
                    if not self.ekf_on:
                        self.notification = 'SLAM is running'
                        self.ekf_on = True
                    else:
                        self.notification = '> 2 landmarks is required for pausing'
                elif n_observed_markers < 3:
                    self.notification = '> 2 landmarks is required for pausing'
                else:
                    if not self.ekf_on:
                        self.request_recover_robot = True
                    self.ekf_on = not self.ekf_on
                    if self.ekf_on:
                        self.notification = 'SLAM is running'
                    else:
                        self.notification = 'SLAM is paused'
            # run object detector
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_p:
                self.command['inference'] = True
            # save object detection outputs
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_n:
                self.command['save_inference'] = True

            elif event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                self.low_speed_mode = True

            # quit
            elif event.type == pygame.QUIT:
                self.quit = True
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                self.quit = True

                # # if in the 2nd stage (CV, mode = 2)
                # if args.mode == 2:
                #     # estimate position of fruits, save it as targets.txt
                #     get_fruit_pose_RF()
                #     # tranform the fruits position
                #     self.gen_est_fruit("lab_output/slam.txt", "lab_output/targets.txt", "aruco_map.txt")

        if self.quit:
            pygame.quit()
            sys.exit()
    


    # transform ARUCO position, save it as aruco_map.txt
    def gen_est_map(self, fname, f_name_est_map):
        with open(fname, 'r') as f:
            try:
                usr_dict = json.load(f)              
            except ValueError as e:
                with open(fname, 'r') as f:
                    usr_dict = ast.literal_eval(f.readline()) 
            aruco_dict = {}
            for (i, tag) in enumerate(usr_dict["taglist"]):
                aruco_dict[tag] = np.reshape([usr_dict["map"][0][i],usr_dict["map"][1][i]], (2,1))
            final_robot_dict = np.reshape([usr_dict["final_robot_pose"][0],usr_dict["final_robot_pose"][1],usr_dict["final_robot_pose"][2]], (3,1))
            
            us_vec =[[], []]
            for i in range(1,11):
                us_vec[0].append(aruco_dict[i][0][0])
                us_vec[1].append(aruco_dict[i][1][0])


        # Finding the transformation of the map
        x,y,rad = 0.0,0.0,0.0
        x = input("X coordinate of the robot at final position: ")
        try:
            x = float(x)
        except ValueError:
            print("Please enter a number.")
            # continue
        y = input("Y coordinate of the robot at final position: ")
        try:
            y = float(y)
        except ValueError:
            print("Please enter a number.")
            # continue
        rad = input("Angle (rad) of the robot at final position: ")
        try:
            rad = float(rad)
        except ValueError:
            print("Please enter a number.")
            # continue

        real_final_pose =  np.hstack([x, y, rad])
        final_robot_dict = final_robot_dict.flatten()

        # points1 - robot real pose, points2 - robot pose from slamstate
        theta = real_final_pose[2] - final_robot_dict[2]

        c, s = np.cos(theta), np.sin(theta)
        R = np.array(((c, -s), (s, c)))
        x = np.array([[real_final_pose[0] ],[real_final_pose[1]]]) - R @ np.array([[final_robot_dict[0]],[final_robot_dict[1]]])
        
        # apply tranforms
        points_transformed =  R @ us_vec + x
        
        map_dict = {}
        for i in range(1,11):
            map_dict["aruco" + str(i) +"_0"] = {"x": points_transformed[0][i-1],"y": points_transformed[1][i-1]}

        with open(f_name_est_map, 'w') as map_f:
            json.dump(map_dict, map_f, indent=2)
    
    
    
    # tranform fruits position, append it to aruco_map.txt
    def gen_est_fruit(self, est_map_fname, targets_fname, output_fname):
        with open(est_map_fname, 'r') as f:
            try:
                usr_dict = json.load(f)              
            except ValueError as e:
                with open(est_map_fname, 'r') as f:
                    usr_dict = ast.literal_eval(f.readline()) 
            aruco_dict = {}
            final_robot_dict = np.reshape([usr_dict["final_robot_pose"][0],usr_dict["final_robot_pose"][1],usr_dict["final_robot_pose"][2]], (3,1))

        with open(targets_fname, 'r') as f:
            try:
                t_dict = json.load(f)                  
            except ValueError as e:
                with open(targets_fname, 'r') as f:
                    t_dict = ast.literal_eval(f.readline()) 
            trgt_dict = {}
            for frt in t_dict.keys():
                trgt_dict[frt[:-2]] = np.reshape([t_dict[frt]["x"],t_dict[frt]["y"]], (2,1))

        fruit_vec =[[], []]
        for i in trgt_dict.keys():
            fruit_vec[0].append(trgt_dict[i][0][0])
            fruit_vec[1].append(trgt_dict[i][1][0])
        
        # Finding the transformation of the map
        x,y,rad = 0.0,0.0,0.0
        x = input("X coordinate of the robot at final position: ")
        try:
            x = float(x)
        except ValueError:
            print("Please enter a number.")
            # continue
        y = input("Y coordinate of the robot at final position: ")
        try:
            y = float(y)
        except ValueError:
            print("Please enter a number.")
            # continue
        rad = input("Angle (rad) of the robot at final position: ")
        try:
            rad = float(rad)
        except ValueError:
            print("Please enter a number.")
            # continue

        real_final_pose =  np.hstack([x, y, rad])
        final_robot_dict = final_robot_dict.flatten()

        # points1 - robot real pose, points2 - robot pose from slamstate
        theta = real_final_pose[2] - final_robot_dict[2]

        c, s = np.cos(theta), np.sin(theta)
        R = np.array(((c, -s), (s, c)))
        x = np.array([[real_final_pose[0] ],[real_final_pose[1]]]) - R @ np.array([[final_robot_dict[0]],[final_robot_dict[1]]])
        
        # apply tranforms
        points_transformed =  R @ fruit_vec + x


        with open(output_fname, 'r') as f:
            try:
                est_dict = json.load(f)                  
            except ValueError as e:
                with open(targets_fname, 'r') as f:
                    est_dict = ast.literal_eval(f.readline()) 
        for idx, i in enumerate(t_dict.keys()):
            est_dict[i] = {"x": points_transformed[0][idx],"y": points_transformed[1][idx]}

        with open(output_fname, 'w') as map_f:
            json.dump(est_dict, map_f, indent=2)

        
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", metavar='', type=str, default='localhost')
    parser.add_argument("--port", metavar='', type=int, default=8000)
    parser.add_argument("--calib_dir", type=str, default="calibration/param/")
    parser.add_argument("--save_data", action='store_true')
    parser.add_argument("--play_data", action='store_true')
    # parser.add_argument("--ckpt", default='network/scripts/model/model.best.pth')
    parser.add_argument("--ckpt", default='network/scripts/model/YOLObest.pt')

    # mode 1: SLAM
    # mode 2: CV
    # parser.add_argument("--mode", metavar='', type=int, default=1)
    args, _ = parser.parse_known_args()
    
    pygame.font.init() 
    TITLE_FONT = pygame.font.Font('pics/8-BitMadness.ttf', 35)
    TEXT_FONT = pygame.font.Font('pics/8-BitMadness.ttf', 40)
    
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

    start = False

    counter = 40
    while not start:
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                start = True
        canvas.blit(splash, (0, 0))
        x_ = min(counter, 600)
        if x_ < 600:
            canvas.blit(pibot_animate[counter%10//2], (x_, 565))
            pygame.display.update()
            counter += 2

    operate = Operate(args)

    
    # # CV mode (identify fruits position)
    # if args.mode == 2:
    #     # load estimated_map.txt
    #     fname = 'aruco_map.txt'
    #     with open(fname, 'r') as f:
    #         try:
    #             aruco_map_dict = json.load(f)                   
    #         except ValueError as e:
    #             with open(fname, 'r') as f:
    #                 aruco_map_dict = ast.literal_eval(f.readline())
        
    #     # convert into a list of [x, y]
    #     est_aruco_pos = []
    #     for i in range(1, 11):
    #         est_aruco_pos.append([ aruco_map_dict[f'aruco{i}_0']['x'], aruco_map_dict[f'aruco{i}_0']['y'] ])
        
    #     # est_map_dict = {}
    #     # for i in range(1, 11):
    #     #     est_map_dict[f"aruco{i}_0"] = {
    #     #         "x": slam_txt_dict['map'][0][slam_txt_dict['taglist'].index(i)],
    #     #         "y": slam_txt_dict['map'][1][slam_txt_dict['taglist'].index(i)]
    #     #         }
        
    #     # Initialise SLAM states with marker position from estimated true map
    #     operate.ekf.freeze_markers = True
    #     LMS = []
    #     for idx, marker in enumerate(est_aruco_pos):
    #         LMS.append(measure.Marker(np.expand_dims(est_aruco_pos[idx], -1), idx+1))
    #     operate.ekf.add_landmarks(LMS)
    #     operate.draw(canvas)
    #     pygame.display.update()
    #     # operate.ekf_on = True
    #     # print(est_aruco_pos)


    while start:
        operate.update_keyboard()
        operate.take_pic()
        drive_meas = operate.control()
        operate.update_slam(drive_meas)
        operate.record_data()
        operate.save_image()
        operate.detect_target()
        # visualise
        operate.draw(canvas)
        pygame.display.update()




