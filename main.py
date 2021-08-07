from plugins import Plugin
import cv2 as cv

if __name__ == '__main__':
    cap = cv.VideoCapture(2)
    p = Plugin()

    print(cap.get(3), cap.get(4))
    while True:
        ret, frame = cap.read()

        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        p.execute(frame)
