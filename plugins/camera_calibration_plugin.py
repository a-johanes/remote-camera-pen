import cv2 as cv
import numpy as np

from plugins.base_plugin import BasePlugin
from plugins.constants import DATA_DIR, MAIN_WINDOW_NAME, KEY_WAIT_DURATION

CAMERA_FILE_NAME = 'camera'
OBJECT_COORDINATES_FILE_NAME = 'coordinates'

# camera calibration
DIMENSIONS = (4, 11)
BLOB_DIST = 26
SNAPSHOT_REQUIRED = 1


class CameraCalibrationPlugin(BasePlugin):
    def __init__(self, print_when_finish) -> None:
        super().__init__(print_when_finish)
        print('START CAMERA PLUGIN')
        self.object_points = []
        self.image_points = []
        self.snapshot_counter = 0
        self.camera_val = {}

        self.__init_object_points__()
        self.__try_load__()

    def __del__(self):
        cv.destroyAllWindows()

    def __init_object_points__(self):
        try:
            print('TRY LOAD COORDINATES')
            with np.load(DATA_DIR + f'{OBJECT_COORDINATES_FILE_NAME}.npz') as coordinates:
                self.object_points = coordinates['arr_0']
        except FileNotFoundError:
            print('INIT NEW COORDINATES')
            self.object_points = np.zeros((DIMENSIONS[0] * DIMENSIONS[1], 3), np.float32)
            for i in range(DIMENSIONS[1]):
                for j in range(0, DIMENSIONS[0]):
                    x = i * BLOB_DIST
                    y = (2 * j + i % 2) * BLOB_DIST
                    z = 0
                    self.object_points[i * DIMENSIONS[0] + j] = np.array([x, y, z])
            np.savez(DATA_DIR + OBJECT_COORDINATES_FILE_NAME, self.object_points)
        print('COORDINATES LOADED')

    def __try_load__(self):
        print('TRY LOAD CAMERA VALUES')
        try:
            with np.load(DATA_DIR + f'{CAMERA_FILE_NAME}.npz') as camera_values:
                mtx, dist, new_cam_mtx, roi, image_points = [camera_values[i] for i in
                                                             ('mtx', 'dist', 'new_cam_mtx', 'roi', 'img_pts')]

            self.image_points = np.array(image_points)

            self.camera_val = {
                'mtx': mtx,
                'dist': dist,
                'new_cam_mtx': new_cam_mtx,
                'roi': roi
            }

            self.finish = True
            print(self.print_when_finish)
        except FileNotFoundError:
            pass

    def __save_camera_value__(self, frame_shape):
        print('SAVING CAMERA VALUE')
        self.snapshot_counter = 0
        object_points = np.array([self.object_points] * SNAPSHOT_REQUIRED)
        self.image_points = np.array(self.image_points)

        ret, mtx, dist, r_vec, t_vec = cv.calibrateCamera(object_points, self.image_points,
                                                          frame_shape[1::-1], None, None)

        h, w = frame_shape[:2]
        new_cam_mtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

        np.savez(DATA_DIR + CAMERA_FILE_NAME,
                 mtx=mtx,
                 new_cam_mtx=new_cam_mtx,
                 roi=roi,
                 dist=dist,
                 img_pts=self.image_points)

        self.camera_val = {
            'mtx': mtx,
            'dist': dist,
            'new_cam_mtx': new_cam_mtx,
            'roi': roi
        }

    def execute(self, frame):
        if self.finish:
            return

        if self.snapshot_counter >= SNAPSHOT_REQUIRED:
            self.__save_camera_value__(frame.shape)
            self.finish = True
            print(self.print_when_finish)
            return

        gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        found, corners = cv.findCirclesGrid(gray_frame, DIMENSIONS,
                                            flags=cv.CALIB_CB_ASYMMETRIC_GRID + cv.CALIB_CB_CLUSTERING)
        frame = cv.drawChessboardCorners(frame, DIMENSIONS, corners, found)

        cv.imshow(MAIN_WINDOW_NAME, frame)
        key = cv.waitKey(KEY_WAIT_DURATION)

        if key == ord('s') and found:
            self.snapshot_counter += 1
            self.image_points.append(corners)
            print(
                f'current snapshot counter: {self.snapshot_counter}/{SNAPSHOT_REQUIRED}')
