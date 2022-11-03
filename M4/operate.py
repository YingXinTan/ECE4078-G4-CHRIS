# basic python packages
import numpy as np
import cv2
import os, sys
import time

# import utility functions
# sys.path.insert(0, "{}/utility".format(os.getcwd()))
# from util.pibot import Alphabot # access the robot
import util.DatasetHandler as dh # save/load functions
# import util.measure as measure # measurements
import pygame # python package for GUI
# import shutil # python package for file operations

# import SLAM components
sys.path.insert(0, "{}/slam".format(os.getcwd()))
from slam.ekf import EKF
from slam.robot import Robot
import slam.aruco_detector as aruco

# import detector
# from network.scripts.detector import Detector

class Operate:
    def __init__(self, args, pibot):
        # self.folder = 'pibot_dataset/'
        # if not os.path.exists(self.folder):
        #     os.makedirs(self.folder)
        # else:
        #     shutil.rmtree(self.folder)
        #     os.makedirs(self.folder)

        # # initialise data parameters
        # if args.play_data:
        #     self.pibot = dh.DatasetPlayer("record")
        # else:
        #     self.pibot = Alphabot(args.ip, args.port)
        self.pibot = pibot

        # initialise SLAM parameters
        self.ekf = self.init_ekf(args.calib_dir, args.ip)
        self.aruco_det = aruco.aruco_detector(self.ekf.robot, marker_length = 0.06)

        if args.save_data:
            self.data = dh.DatasetWriter('record')
        else:
            self.data = None
        # self.output = dh.OutputWriter('lab_output')
        self.command = {'inference': False,
                        'save_inference': False}
        # self.command = {'motion':[0, 0],
        #                 'inference': False,
        #                 'output': False,
        #                 'save_inference': False,
        #                 'save_image': False}
        # self.quit = False
        # self.pred_fname = ''
        self.request_recover_robot = False
        # self.file_output = None
        self.ekf_on = False
        # self.double_reset_comfirm = 0
        # self.image_id = 0
        self.notification = 'Milestone 4'
        
        # a 5min timer
        # self.count_down = 300
        # self.start_time = time.time()
        # self.control_clock = time.time()

        # initialise images
        self.img = np.zeros([240,320,3], dtype=np.uint8)
        self.aruco_img = np.zeros([240,320,3], dtype=np.uint8)

        # initialise fruit detector output
        self.detector_output = np.zeros([240,320], dtype=np.uint8)
        if args.ckpt == "":
            self.detector = None
            self.network_vis = cv2.imread('pics/8bit/detector_splash.png')
        else:
            # self.detector = Detector(args.ckpt, use_gpu=False)
            self.network_vis = np.ones((240, 320,3))* 100

        self.bg = pygame.image.load('pics/gui_mask.jpg')
    
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
                print('Robot pose is successfuly recovered')
                self.ekf_on = True
            else:
                print('Recover failed, need >2 landmarks!')
                self.ekf_on = False
            self.request_recover_robot = False
        elif self.ekf_on: # and not self.debug_flag:
            self.ekf.predict(drive_meas)
            self.ekf.add_landmarks(lms)
            self.ekf.update(lms)
    
    # using computer vision to detect targets
    def detect_target(self):
        if self.command['inference'] and self.detector is not None:
            self.detector_output, self.network_vis = self.detector.detect_single_image(self.img)
            self.command['inference'] = False
            self.file_output = (self.detector_output, self.ekf)
            self.notification = f'{len(np.unique(self.detector_output))-1} target type(s) detected'
    




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
