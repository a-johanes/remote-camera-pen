import math

import cv2 as cv
import numpy as np
import pyautogui

from plugins.base_plugin import BasePlugin
from plugins.color_calibration_plugin import ColorCalibrationPlugin
from plugins.constants import MAIN_WINDOW_NAME, MASK_WINDOW_NAME, KEY_WAIT_DURATION
from plugins.homography_plugin import HomographyPlugin

pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0

CURSOR_POOL = 3
SHADOW_WINDOW_W = 120
SHADOW_WINDOW_H = 60

CURSOR_SPEED_MULTIPLIER = 2
MIN_DELTA_X = 1
MIN_DELTA_Y = 2
MAX_POS_DIV = 10


class PenCapTrackingPlugin(BasePlugin):
    def __init__(self):
        super().__init__('PEN TRACK FINISH')
        print('START PEN CAP TRACK PLUGIN')

        cv.namedWindow(MAIN_WINDOW_NAME)
        cv.namedWindow(MASK_WINDOW_NAME)

        cv.moveWindow(MAIN_WINDOW_NAME, 0, 0)
        cv.moveWindow(MASK_WINDOW_NAME, 0, 640)

        self.homography_plugin = HomographyPlugin('EXECUTING PEN CALIBRATION PLUGIN')
        self.pen_calibration_plugin = ColorCalibrationPlugin('pen', 'EXECUTING SHADOW CALIBRATION PLUGIN')
        self.shadow_calibration_plugin = ColorCalibrationPlugin('shadow', 'EXECUTING PEN TRACK PLUGIN')

        self.plugins = [
            self.homography_plugin,
            self.pen_calibration_plugin,
            self.shadow_calibration_plugin
        ]
        self.curr_plugin_idx = 0

        self.prev_pen_pos = (-1, -1)
        self.cursor_pool_counter = 0
        self.moving_avg = []
        self.click = True

        self.kernel = np.ones((5, 5), np.uint8)

        self.finish = False

    def __del__(self):
        cv.destroyAllWindows()

    def __preprocess_input__(self, frame):
        # cam_val = self.homography_plugin.camera_plugin.camera_val

        # undistorted
        # frame = cv.undistort(frame, cam_val['mtx'], cam_val['dist'], None,
        #                      cam_val['new_cam_mtx'])
        # crop the image
        # x, y, w, h = cam_val['roi']
        # frame = frame[y:y + h, x:x + w]

        # flip
        # frame = cv.flip(frame, 1)

        return frame

    @staticmethod
    def draw_guide(frame, window_top_right_pos, pen_pos, shadow_pos):
        window_top_right_x, window_top_right_y = window_top_right_pos
        pen_x, pen_y = pen_pos
        shadow_x, shadow_y = shadow_pos

        # draw window border
        cv.line(frame,
                (window_top_right_x, window_top_right_y),
                (window_top_right_x, window_top_right_y + SHADOW_WINDOW_H),
                [255, 255, 255])
        cv.line(frame,
                (window_top_right_x, window_top_right_y),
                (window_top_right_x - SHADOW_WINDOW_W, window_top_right_y),
                [255, 255, 255])
        cv.line(frame,
                (window_top_right_x - SHADOW_WINDOW_W, window_top_right_y + SHADOW_WINDOW_H),
                (window_top_right_x - SHADOW_WINDOW_W, window_top_right_y),
                [255, 255, 255])
        cv.line(frame,
                (window_top_right_x - SHADOW_WINDOW_W, window_top_right_y + SHADOW_WINDOW_H),
                (window_top_right_x, window_top_right_y + SHADOW_WINDOW_H),
                [255, 255, 255])

        # draw tip position marker
        cv.circle(frame, (shadow_x, shadow_y), 4, [0, 0, 255], 4)
        cv.circle(frame, (pen_x, shadow_y), 4, [255, 255, 255], 4)
        cv.circle(frame, (pen_x, pen_y), 4, [0, 0, 255], 4)

        return frame

    def __check_mouse__(self, pen_tip_pos, shadow_pos):
        pen_x, pen_y = pen_tip_pos
        shadow_x, shadow_y = shadow_pos

        self.moving_avg.append([pen_x, pen_y, shadow_x, shadow_y])

        if len(self.moving_avg) > CURSOR_POOL:
            self.moving_avg.pop(0)

        self.cursor_pool_counter += 1
        self.cursor_pool_counter %= CURSOR_POOL

        self.__check_click__(pen_tip_pos, shadow_pos)

        if self.cursor_pool_counter != 0:
            return

        self.__move__mouse__()

    def __move__mouse__(self):
        pen_x, pen_y, shadow_x, shadow_y = np.mean(self.moving_avg, axis=0)

        homography_mtx = self.homography_plugin.homography_mtx
        # mouse_point = (pen_x, pen_y)

        # if not self.click:
        mouse_point = (pen_x, shadow_y)

        world_x, world_y, world_w = homography_mtx @ np.array([[*mouse_point, 1]]).T
        screen_x, screen_y = world_x / world_w, world_y / world_w

        if self.prev_pen_pos == (-1, -1):
            self.prev_pen_pos = (screen_x, screen_y)

        dx = -(screen_y - self.prev_pen_pos[1])
        dy = -(screen_x - self.prev_pen_pos[0])

        if abs(dx) <= MIN_DELTA_X:
            dx = 0
        if abs(dy) <= MIN_DELTA_Y:
            dy = 0

        dx, dy = math.floor(dx), math.floor(dy)
        pyautogui.move(dx * CURSOR_SPEED_MULTIPLIER,
                       dy * CURSOR_SPEED_MULTIPLIER)

        self.prev_pen_pos = (screen_x, screen_y)

    def __check_click__(self, pen_tip_pos, shadow_pos):
        pen_x, pen_y = pen_tip_pos
        shadow_x, shadow_y = shadow_pos

        shadow_dist = max(shadow_y, pen_y) - pen_y

        if shadow_dist <= 8:
            if not self.click:
                self.click = True
                pyautogui.mouseDown()
        else:
            if shadow_dist >= 13 and self.click:
                self.click = False
                pyautogui.mouseUp()

    def execute(self, frame):
        while self.curr_plugin_idx < len(self.plugins) and self.plugins[self.curr_plugin_idx].finish:
            self.curr_plugin_idx += 1

        if self.curr_plugin_idx < len(self.plugins):
            self.plugins[self.curr_plugin_idx].execute(frame)
            return

        # all required plugins already finish

        frame = self.__preprocess_input__(frame)

        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

        pen_hsv = self.pen_calibration_plugin.hsv_value
        pen_mask = cv.inRange(hsv, pen_hsv['low'], pen_hsv['high'])
        pen_mask = cv.erode(pen_mask, self.kernel, iterations=1)
        pen_mask = cv.dilate(pen_mask, self.kernel, iterations=2)

        pen_contours, _ = cv.findContours(pen_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        if not pen_contours:
            self.prev_pen_pos = (-1, -1)
            cv.imshow(MAIN_WINDOW_NAME, frame)
            cv.waitKey(KEY_WAIT_DURATION)
            return

        pen_max_contour = max(pen_contours, key=cv.contourArea)
        # pen_contour_area = cv.contourArea(pen_maks_contour)

        # simplify the contour
        epsilon = 0.1 * cv.arcLength(pen_max_contour, True)
        simplified_contour = cv.approxPolyDP(pen_max_contour, epsilon, True)

        # find biggest y point (lowest)
        [[_, idx]] = np.argmax(simplified_contour, axis=0)
        pen_tip_pos = simplified_contour[idx][0]

        window_top_right_x = pen_tip_pos[0] + SHADOW_WINDOW_W // 3
        window_top_right_y = pen_tip_pos[1] - SHADOW_WINDOW_H // 3

        shadow_window = hsv[window_top_right_y: window_top_right_y + SHADOW_WINDOW_H,
                        window_top_right_x - SHADOW_WINDOW_W: window_top_right_x]

        shadow_threshold = SHADOW_WINDOW_W * SHADOW_WINDOW_H * 255 // 25

        # check if frame is empty (no shadow detected)
        if shadow_window.size:
            # only find shadow around pen
            shadow_hsv = self.shadow_calibration_plugin.hsv_value
            shadow_mask = cv.inRange(shadow_window, shadow_hsv['low'], shadow_hsv['high'])
            cv.imshow(MASK_WINDOW_NAME, shadow_mask)

            shadow_contours, _ = cv.findContours(shadow_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

            if shadow_contours and shadow_mask.sum() >= shadow_threshold:

                shadow_max_contour = max(shadow_contours, key=cv.contourArea)

                # find the right most position
                [[curr_idx, _]] = np.argmax(shadow_max_contour, axis=0)
                shadow_x, shadow_y = shadow_max_contour[curr_idx][0]

                shadow_x += window_top_right_x - SHADOW_WINDOW_W
                shadow_y += window_top_right_y

                frame = PenCapTrackingPlugin.draw_guide(frame, (window_top_right_x, window_top_right_y), pen_tip_pos,
                                                        (shadow_x, shadow_y))

                self.__check_mouse__(pen_tip_pos, (shadow_x, shadow_y))

            else:
                self.prev_pen_pos = (-1, -1)
                self.moving_avg = []
        else:
            self.prev_pen_pos = (-1, -1)
            self.moving_avg = []

        cv.imshow(MAIN_WINDOW_NAME, frame)
        cv.waitKey(KEY_WAIT_DURATION)
