import cv2 as cv
import numpy as np

from plugins.base_plugin import BasePlugin
from plugins.camera_calibration_plugin import CameraCalibrationPlugin
from plugins.constants import DATA_DIR, MAIN_WINDOW_NAME

HOMOGRAPHY_FILE_NAME = 'homography'


class HomographyPlugin(BasePlugin):
    def __init__(self, print_when_finish):
        super().__init__(print_when_finish)
        print('START HOMOGRAPHY PLUGIN')
        self.homography_mtx = None
        self.camera_plugin = CameraCalibrationPlugin('EXECUTING HOMOGRAPHY CALIBRATION PLUGIN')
        self.__try_load__()

    def __del__(self):
        cv.destroyAllWindows()

    def __try_load__(self):
        print('TRY LOAD HOMOGRAPHY MATRIX')

        try:
            with np.load(DATA_DIR + f'{HOMOGRAPHY_FILE_NAME}.npz') as homography_mtx:
                self.homography_mtx = homography_mtx['arr_0']
            self.finish = True
        except FileNotFoundError:
            if not self.camera_plugin.finish:
                return

            image_points = self.camera_plugin.image_points[0].reshape(-1, 2)
            object_points = self.camera_plugin.object_points
            self.homography_mtx, status = cv.findHomography(image_points, object_points)
            np.savez(DATA_DIR + HOMOGRAPHY_FILE_NAME, self.homography_mtx)

            self.finish = True
            print(self.print_when_finish)

    def execute(self, frame):
        if self.finish and self.camera_plugin.finish:
            return

        if not self.camera_plugin.finish:
            self.camera_plugin.execute(frame)
            return

        image_points = self.camera_plugin.image_points[0].reshape(-1, 2)
        object_points = self.camera_plugin.object_points
        self.homography_mtx, status = cv.findHomography(image_points, object_points)
        np.savez(DATA_DIR + HOMOGRAPHY_FILE_NAME, self.homography_mtx)

        self.finish = True
        print(self.print_when_finish)
