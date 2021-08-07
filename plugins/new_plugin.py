from enum import Enum

import cv2 as cv
import numpy as np
import pyautogui

pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0

mouse_now = pyautogui.position()

SCREEN_SIZE = pyautogui.size()
CURSOR_SPEED_MULTIPLIER = 2
# MAX_DELTA = 20
MIN_DELTA_X = 1
MIN_DELTA_Y = 2
CURSOR_POOL = 5

KEY_WAIT_DURATION = 1

CALIBRATION_DIMENSIONS = (4, 11)
BLOB_DIST = 26
CAMERA_CALIBRATION_REQUIRED = 1

MAIN_WINDOW_NAME = 'main'
MASK_WINDOW_NAME = 'mask'

DATA_DIRECTORY = './data/'


class PluginStatus(Enum):
    TRY_LOAD_CALIBRATE_CAMERA = 1
    CALIBRATE_CAMERA = 2
    TRY_LOAD_HOMOGRAPHY_MTX = 3
    CALCULATE_HOMOGRAPHY_MTX = 4
    TRY_LOAD_CALIBRATE_PEN = 5
    CALIBRATE_PEN = 6
    TRY_LOAD_CALIBRATE_SHADOW = 7
    CALIBRATE_SHADOW = 8
    READY = 9


class Plugin:
    def __init__(self):
        self.real_world_point_coordinate = np.zeros((CALIBRATION_DIMENSIONS[0] * CALIBRATION_DIMENSIONS[1], 3),
                                                    np.float32)
        self.image_points = []
        self.cursor_pool_counter = 0

        self.boundary = {
            'x': (0, 0),
            'y': (0, 0)
        }

        self.camera_calibration_counter = 0
        self.homography_mtx = None

        self.prev_pen_pos = (-1, -1)

        self.curr_frame = None
        self.curr_hsv_list = []
        self.status = PluginStatus.TRY_LOAD_CALIBRATE_CAMERA

        self.camera_values = {}

        self.pen_hsv = {'high': np.array([0, 0, 0]), 'low': np.array([180, 255, 255])}
        self.shadow_hsv = {'high': np.array([0, 0, 0]), 'low': np.array([180, 255, 255])}

        print(self.status)

        self.estimate_real_world_coordinate()

        cv.namedWindow(MAIN_WINDOW_NAME)
        cv.namedWindow(MASK_WINDOW_NAME)

        cv.moveWindow(MAIN_WINDOW_NAME, 0, 0)
        cv.moveWindow(MASK_WINDOW_NAME, 1000, 0)

        cv.setMouseCallback(MAIN_WINDOW_NAME, self.calibrate_hsv_callback)

        self.sld = []
        self.click = False

    def estimate_real_world_coordinate(self):
        try:
            with np.load(DATA_DIRECTORY + 'coordinates.npz') as coordinates:
                self.real_world_point_coordinate = coordinates['arr_0']
        except FileNotFoundError:
            for i in range(CALIBRATION_DIMENSIONS[1]):
                for j in range(0, CALIBRATION_DIMENSIONS[0]):
                    x = i * BLOB_DIST
                    y = (2 * j + i % 2) * BLOB_DIST
                    z = 0
                    self.real_world_point_coordinate[i * CALIBRATION_DIMENSIONS[0] + j] = np.array([x, y, z])
            np.savez(DATA_DIRECTORY + 'coordinates', self.real_world_point_coordinate)

    def try_load_calibrate_camera(self):
        try:
            with np.load(DATA_DIRECTORY + 'camera.npz') as camera_value:
                mtx, dist, new_camera_mtx, roi, image_points = [camera_value[i] for i in
                                                                ('mtx', 'dist', 'new_camera_mtx', 'roi', 'img_pts')]

            self.image_points = np.array(image_points).reshape(-1, 2)
            boundary_min, boundary_max = np.amin(self.image_points, axis=0), np.amax(self.image_points, axis=0)
            self.boundary = {
                'x': (int(boundary_min[0]), int(boundary_max[0])),
                'y': (int(boundary_min[1]), int(boundary_max[1]))
            }

            self.camera_values = {
                'mtx': mtx,
                'dist': dist,
                'new_camera_mtx': new_camera_mtx,
                'roi': roi
            }
            self.status = PluginStatus(self.status.value + 2)
            print(self.status)

        except FileNotFoundError:
            self.status = PluginStatus(self.status.value + 1)
            print(self.status)

    def calibrate_camera(self, frame):
        if self.camera_calibration_counter >= CAMERA_CALIBRATION_REQUIRED:
            self.camera_calibration_counter = 0
            object_points = np.array([self.real_world_point_coordinate] * CAMERA_CALIBRATION_REQUIRED)

            ret, mtx, dist, r_vec, t_vec = cv.calibrateCamera(object_points, self.image_points,
                                                              self.curr_frame.shape[1::-1], None, None)

            h, w = frame.shape[:2]
            print(h, w)
            new_camera_mtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

            np.savez(DATA_DIRECTORY + 'camera',
                     mtx=mtx,
                     new_camera_mtx=new_camera_mtx,
                     roi=roi,
                     dist=dist,
                     img_pts=self.image_points)

            self.camera_values = {
                'mtx': mtx,
                'dist': dist,
                'new_camera_mtx': new_camera_mtx,
                'roi': roi
            }

            self.image_points = np.array(self.image_points).reshape(-1, 2)
            boundary_min, boundary_max = np.amin(self.image_points, axis=0), np.amax(self.image_points, axis=0)
            self.boundary = {
                'x': (int(boundary_min[0]), int(boundary_max[0])),
                'y': (int(boundary_min[1]), int(boundary_max[1]))
            }
            print(self.boundary)

            self.status = PluginStatus(self.status.value + 1)
            print(self.status)
            return

        self.curr_frame = frame.copy()
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        ret, corners = cv.findCirclesGrid(gray, CALIBRATION_DIMENSIONS,
                                          flags=cv.CALIB_CB_ASYMMETRIC_GRID + cv.CALIB_CB_CLUSTERING)
        frame = cv.drawChessboardCorners(frame, CALIBRATION_DIMENSIONS, corners, ret)

        cv.imshow(MAIN_WINDOW_NAME, frame)
        key = cv.waitKey(KEY_WAIT_DURATION)

        if key == ord('s') and ret:
            self.camera_calibration_counter += 1
            self.image_points.append(corners)
            print(
                f'current camera calibration counter: {self.camera_calibration_counter}/{CAMERA_CALIBRATION_REQUIRED}')

    def try_load_homography_mtx(self):
        try:
            with np.load(DATA_DIRECTORY + 'homography.npz') as homography_mtx:
                self.homography_mtx = homography_mtx['arr_0']

            self.status = PluginStatus(self.status.value + 2)
            print(self.status)

        except FileNotFoundError:
            self.status = PluginStatus(self.status.value + 1)
            print(self.status)

    def calculate_homography_mtx(self):
        image_points = np.array(self.image_points).reshape((-1, 2))
        object_points = np.array([self.real_world_point_coordinate] * CAMERA_CALIBRATION_REQUIRED)
        object_points = object_points.reshape((-1, 3))

        self.homography_mtx, status = cv.findHomography(image_points, object_points)
        np.savez(DATA_DIRECTORY + 'homography', self.homography_mtx)

        self.status = PluginStatus(self.status.value + 1)
        print(self.status)

    def calibrate_hsv_callback(self, event, x, y, flags, user_data):
        if event != cv.EVENT_LBUTTONDOWN:
            return

        hsv_image = cv.cvtColor(self.curr_frame, cv.COLOR_BGR2HSV)
        hsv_point = hsv_image[y, x]
        self.curr_hsv_list += [hsv_point * 0.8, hsv_point * 1.2]

    def try_load_calibrate_hsv(self):
        try:
            if self.status == PluginStatus.TRY_LOAD_CALIBRATE_PEN:
                with np.load(DATA_DIRECTORY + 'pen_hsv.npz') as temp_hsv:
                    self.pen_hsv['high'] = temp_hsv['high']
                    self.pen_hsv['low'] = temp_hsv['low']
            else:
                with np.load(DATA_DIRECTORY + 'shadow_hsv.npz') as temp_hsv:
                    self.shadow_hsv['high'] = temp_hsv['high']
                    self.shadow_hsv['low'] = temp_hsv['low']

            self.status = PluginStatus(self.status.value + 2)
            print(self.status)
        except FileNotFoundError:
            self.status = PluginStatus(self.status.value + 1)
            print(self.status)

    def calibrate_hsv(self, frame):
        self.curr_frame = frame

        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        if self.curr_hsv_list:
            mask = cv.inRange(hsv, np.amin(self.curr_hsv_list, axis=0), np.amax(self.curr_hsv_list, axis=0))
            cv.imshow(MASK_WINDOW_NAME, mask)

        cv.imshow(MAIN_WINDOW_NAME, frame)

        key = cv.waitKey(KEY_WAIT_DURATION)

        if key == ord('s'):
            # save hsv
            low_hsv = np.amin(self.curr_hsv_list, axis=0)
            high_hsv = np.amax(self.curr_hsv_list, axis=0)
            name = 'pen_hsv' if self.status == PluginStatus.CALIBRATE_PEN else 'shadow_hsv'
            np.savez(DATA_DIRECTORY + name, low=low_hsv, high=high_hsv)

            if self.status == PluginStatus.CALIBRATE_PEN:
                self.pen_hsv['low'] = low_hsv
                self.pen_hsv['high'] = high_hsv
            else:
                self.shadow_hsv['low'] = low_hsv
                self.shadow_hsv['high'] = high_hsv

            self.curr_hsv_list = []
            self.status = PluginStatus(self.status.value + 1)
            print(self.status)

        if key == ord('r'):
            # rest hsv
            self.curr_hsv_list = []

    def track(self, frame):
        kernel = np.ones((5, 5), np.uint8)

        # # undistorted
        # frame = cv.undistort(frame, self.camera_values['mtx'], self.camera_values['dist'], None,
        #                      self.camera_values['new_camera_mtx'])
        # # crop the image
        # x, y, w, h = self.camera_values['roi']
        # frame = frame[y:y + h, x:x + w]

        # frame = cv.flip(frame, 1)

        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

        pen_mask = cv.inRange(hsv, self.pen_hsv['low'], self.pen_hsv['high'])
        pen_mask = cv.erode(pen_mask, kernel, iterations=1)
        pen_mask = cv.dilate(pen_mask, kernel, iterations=2)

        pen_contours, _ = cv.findContours(pen_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        if not pen_contours:
            self.prev_pen_pos = (-1, -1)
            cv.imshow(MAIN_WINDOW_NAME, frame)
            cv.waitKey(KEY_WAIT_DURATION)
            return

        pen_maks_contour = max(pen_contours, key=cv.contourArea)
        # pen_contour_area = cv.contourArea(pen_maks_contour)

        # find biggest y point (lowest)
        # hull = cv.convexHull(pen_maks_contour)
        epsilon = 0.1 * cv.arcLength(pen_maks_contour, True)
        approx = cv.approxPolyDP(pen_maks_contour, epsilon, True)

        [[_, idx]] = np.argmax(approx, axis=0)
        pen_x, pen_y = approx[idx][0]

        # if self.prev_pen_pos == (-1, -1):
        #     self.prev_pen_pos = (pen_x, pen_y)

        # if self.prev_pen_pos == (-1, -1):
        #     self.prev_pen_pos = (screen_x, screen_y)
        #     return

        window_w = 120
        window_h = 60

        window_top_right_x = pen_x + window_w // 3
        window_top_right_y = pen_y - window_h // 3

        # window_top_right_x = self.prev_pen_pos[0] + window_w // 3
        # window_top_right_y = self.prev_pen_pos[1] - window_h // 3

        window = hsv[window_top_right_y:window_top_right_y + window_h, window_top_right_x - window_w:window_top_right_x]
        cv.line(frame, (window_top_right_x, window_top_right_y), (window_top_right_x, window_top_right_y + window_h),
                [255, 255, 255])
        cv.line(frame, (window_top_right_x, window_top_right_y), (window_top_right_x - window_w, window_top_right_y),
                [255, 255, 255])
        cv.line(frame, (window_top_right_x - window_w, window_top_right_y + window_h),
                (window_top_right_x - window_w, window_top_right_y), [255, 255, 255])
        cv.line(frame, (window_top_right_x - window_w, window_top_right_y + window_h),
                (window_top_right_x, window_top_right_y + window_h), [255, 255, 255])

        shadow_threshold = window_w * window_h * 255 // 50

        if window.size:
            # only find shadow around pen
            shadow_mask = cv.inRange(window, self.shadow_hsv['low'], self.shadow_hsv['high'])
            cv.imshow(MASK_WINDOW_NAME, shadow_mask)
            # shadow_frame = cv.bitwise_and(frame, frame, mask=shadow_mask)
            shadow_contours, _ = cv.findContours(shadow_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

            # cv.putText(frame, f'{shadow_mask.sum() / (window_h * window_w * 255):.2f}', (pen_x + 10, pen_y + 30),
            #            cv.FONT_HERSHEY_SIMPLEX, 1, [0, 255, 0])

            if shadow_contours and shadow_mask.sum() >= shadow_threshold:

                shadow_maks_contour = max(shadow_contours, key=cv.contourArea)
                # shadow_contour_area = cv.contourArea(shadow_maks_contour)
                # shadow_y, shadow_x = frame.shape[:2]
                [[curr_idx, _]] = np.argmax(shadow_maks_contour, axis=0)
                shadow_x, shadow_y = shadow_maks_contour[curr_idx][0]

                # shadow_x, shadow_y = 0, 0
                #
                # for contour in shadow_contours:
                #     [[curr_idx, _]] = np.argmax(contour, axis=0)
                #     curr_x, curr_y = contour[curr_idx][0]
                #     if curr_x > shadow_x:
                #         shadow_x = curr_x
                #         shadow_y = curr_y

                estimated_contact_pos_x, estimated_contact_pos_y = pen_x, shadow_y
                shadow_x += window_top_right_x - window_w
                shadow_y += window_top_right_y

                # shadow_dist = dist((pen_x, pen_y), (estimated_contact_pos_x, estimated_contact_pos_y))
                shadow_dist = shadow_y - pen_y
                # shadow_dist = np.std(shadow_mask) / np.mean(shadow_mask) > 2 and dist((pen_x, pen_y),
                #                                                                       (shadow_x, shadow_y)) < 10
                # if self.sld:
                # cv.putText(frame, f'{np.mean(self.sld):.2f}, {np.argmax(self.sld):.2f}', (shadow_x, shadow_y + 30),
                #            cv.FONT_HERSHEY_SIMPLEX, 1,
                #            [0, 255, 0])

                cv.circle(frame, (shadow_x, shadow_y), 4, [0, 0, 255], 4)
                cv.circle(frame, (pen_x, shadow_y), 4, [255, 255, 255], 4)

                self.move_mouse(pen_x, pen_y, shadow_x, shadow_y)
                cv.putText(frame, f'{pen_x}, {pen_y}', (pen_x, pen_y - 30), cv.FONT_HERSHEY_SIMPLEX, 1,
                           [0, 255, 0])
            else:
                self.prev_pen_pos = (-1, -1)

        else:
            self.prev_pen_pos = (-1, -1)


        # [[x1,y1,w]] = (homo @ np.array([[x,y,1]]).T).T
        cv.circle(frame, (pen_x, pen_y), 4, [0, 0, 255], 4)
        # cv.circle(frame, (shadow_x, shadow_y), int(shadow_dist), [0, 0, 255], 4)
        cv.drawContours(frame, approx, -1, [0, 255, 0], 4)
        # cv.putText(frame, f'{screen_x}, {screen_y}', (pen_x, pen_y - 30), cv.FONT_HERSHEY_SIMPLEX, 1,
        #            [0, 255, 0])

        # scale_percent = 300  # percent of original size
        # width = int(crop.shape[1] * scale_percent / 100)
        # height = int(crop.shape[0] * scale_percent / 100)
        # dim = (width, height)
        # crop_resized = cv.resize(crop, dim)

        # if crop.size:
        #     gray = cv.cvtColor(crop, cv.COLOR_BGR2GRAY)
        #     m = np.mean(gray)
        #     s = np.std(gray)
        #     # print(s/m)
        #     if s/m > 1.5:
        #         if not click:
        #             click = True
        #             counter += 1
        #             print('click', counter)
        #     else:
        #         click = False
        #
        #     cv.imshow('frame', crop)

        cv.imshow(MAIN_WINDOW_NAME, frame)
        cv.waitKey(KEY_WAIT_DURATION)

    def move_mouse(self, pen_x, pen_y, shadow_x, shadow_y):
        self.sld.append([pen_x, pen_y, shadow_x, shadow_y])

        if len(self.sld) > CURSOR_POOL:
            self.sld.pop(0)

        self.cursor_pool_counter += 1
        self.cursor_pool_counter %= CURSOR_POOL

        pen_x, pen_y, shadow_x, shadow_y = np.mean(self.sld, axis=0)

        # shadow_dist = (shadow_y if shadow_x > pen_x else pen_y)  - pen_y
        shadow_dist = shadow_y - pen_y
        if shadow_dist <= 5:
            # world_x, world_y, world_w = self.homography_mtx @ np.array([[pen_x, pen_y, 1]]).T
            # screen_x, screen_y = world_x // world_w, world_y // world_w

            if not self.click:
                self.click = True
                # print('click', shadow_dist)
                pyautogui.mouseDown()
        else:
            # world_x, world_y, world_w = self.homography_mtx @ np.array([[pen_x, shadow_y, 1]]).T
            if shadow_dist >= 13 and self.click:
                self.click = False
                # print('up', shadow_dist)
                pyautogui.mouseUp()

        if self.cursor_pool_counter != 0:
            return

        if self.click:
            world_x, world_y, world_w = self.homography_mtx @ np.array([[pen_x, pen_y, 1]]).T
        else:
            world_x, world_y, world_w = self.homography_mtx @ np.array([[pen_x, shadow_y, 1]]).T

        # print(shadow_dist)

        screen_x, screen_y = world_x // world_w, world_y // world_w

        if self.prev_pen_pos == (-1, -1):
            self.prev_pen_pos = (screen_x, screen_y)
        # else:
        #     print(screen_x, screen_y, self.prev_pen_pos[0], self.prev_pen_pos[1])

        # if abs(screen_x - self.prev_pen_pos[0]) > 1:
        #     screen_x = math.floor(self.prev_pen_pos[0] + 0.5)
        #
        # if abs(screen_y - self.prev_pen_pos[1]) > 1:
        #     screen_y = math.floor(self.prev_pen_pos[1] + 0.5)


        # print(dx, dy)

        # if abs(screen_x - self.prev_pen_pos[0]) >= 1:
        #     screen_x = math.floor(self.prev_pen_pos[0] + 0.5)
        #
        # if abs(screen_y - self.prev_pen_pos[1]) >= 1:
        #     screen_y = math.floor(self.prev_pen_pos[1] + 0.5)

        dx = -(screen_y - self.prev_pen_pos[1])
        dy = -(screen_x - self.prev_pen_pos[0])

        if abs(dx) <= MIN_DELTA_X:
            dx = 0
        if abs(dy) <= MIN_DELTA_Y:
            dy = 0

        pyautogui.move(dx * CURSOR_SPEED_MULTIPLIER,
                       dy * CURSOR_SPEED_MULTIPLIER)

        self.prev_pen_pos = (screen_x, screen_y)
        # return dx, dy

    def execute(self, frame):
        if self.status == PluginStatus.TRY_LOAD_CALIBRATE_CAMERA:
            self.try_load_calibrate_camera()
        elif self.status == PluginStatus.CALIBRATE_CAMERA:
            self.calibrate_camera(frame)
        elif self.status == PluginStatus.TRY_LOAD_HOMOGRAPHY_MTX:
            self.try_load_homography_mtx()
        elif self.status == PluginStatus.CALCULATE_HOMOGRAPHY_MTX:
            self.calculate_homography_mtx()
        elif self.status == PluginStatus.TRY_LOAD_CALIBRATE_PEN:
            self.try_load_calibrate_hsv()
        elif self.status == PluginStatus.CALIBRATE_PEN:
            self.calibrate_hsv(frame)
        elif self.status == PluginStatus.TRY_LOAD_CALIBRATE_SHADOW:
            self.try_load_calibrate_hsv()
        elif self.status == PluginStatus.CALIBRATE_SHADOW:
            self.calibrate_hsv(frame)
        elif self.status == PluginStatus.READY:
            self.track(frame)


if __name__ == '__main__':
    cap = cv.VideoCapture(0)
    p = Plugin()
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    while True:
        ret, vid_frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        p.execute(vid_frame)
