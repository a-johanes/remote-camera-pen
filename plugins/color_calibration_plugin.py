import cv2 as cv
import numpy as np

from plugins.base_plugin import BasePlugin
from plugins.constants import DATA_DIR, MASK_WINDOW_NAME, MAIN_WINDOW_NAME, KEY_WAIT_DURATION


class ColorCalibrationPlugin(BasePlugin):
    def __init__(self, tag, print_when_finish):
        super().__init__(print_when_finish)
        print('START COLOR CALIBRATION PLUGIN', tag)
        self.tag = tag  # differentiate between pen and shadow
        self.hsv_value = {}  # store high and low hsv boundary
        self.hsv_list = []  # store all hsv value clicked from image
        self.frame = None  # duplicate of current frame for callback
        self.is_set_callback = False

        self.__try_load__()

    def __del__(self):
        cv.setMouseCallback(MAIN_WINDOW_NAME, None)
        cv.destroyAllWindows()

    def __try_load__(self):
        print('TRY LOAD COLOR VALUE', self.tag)
        try:
            with np.load(DATA_DIR + f'{self.tag}.npz') as temp_hsv:
                self.hsv_value['high'] = temp_hsv['high']
                self.hsv_value['low'] = temp_hsv['low']
            self.finish = True
            print(self.print_when_finish)
        except FileNotFoundError:
            pass

    def __window_callback__(self, event, x, y, flags, user_data):
        if event != cv.EVENT_LBUTTONDOWN:
            return

        if self.frame is None or self.frame.size < 1:
            return

        hsv_frame = cv.cvtColor(self.frame, cv.COLOR_BGR2HSV)
        hsv_point = hsv_frame[y, x]
        self.hsv_list += [hsv_point * 0.8, hsv_point * 1.2]

    def execute(self, frame):
        if self.finish:
            return

        if not self.is_set_callback:
            cv.setMouseCallback(MAIN_WINDOW_NAME, self.__window_callback__)

        self.frame = frame  # for callback purposes

        hsv_frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

        if self.hsv_list:
            mask = cv.inRange(hsv_frame, np.amin(self.hsv_list, axis=0), np.amax(self.hsv_list, axis=0))
            cv.imshow(MASK_WINDOW_NAME, mask)

        cv.imshow(MAIN_WINDOW_NAME, frame)

        key = cv.waitKey(KEY_WAIT_DURATION)

        if key == ord('s'):
            # save hsv
            low_hsv = np.amin(self.hsv_list, axis=0)
            high_hsv = np.amax(self.hsv_list, axis=0)
            np.savez(DATA_DIR + self.tag, low=low_hsv, high=high_hsv)

            self.hsv_value['low'] = low_hsv
            self.hsv_value['high'] = high_hsv

            self.hsv_list = []

            self.finish = True

            print(self.print_when_finish)

        if key == ord('r'):
            # rest hsv
            self.hsv_list = []
