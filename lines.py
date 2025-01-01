import cv2
import numpy as np

cam = cv2.VideoCapture(0)

while True:
    check, frame = cam.read()
    cv2.imshow("input", frame)

    bg_color = np.mean(np.mean(frame, axis=0), axis=0)
    bg = np.ones_like(frame) * bg_color
    cv2.imshow("bg", np.astype(bg, np.uint8))

    diff = np.mean(frame - bg, axis=-1)
    grating = np.array(np.sin(range(frame.shape[0])) > 0)
    white = ((diff > 50).T * grating).T
    black = ((diff < -50).T * (1 -grating)).T
    cv2.imshow("white", white * 1.0)
    cv2.imshow("black", black * 1.0)

    cv2.imshow("res", np.astype(bg * (1 - black[:, :, np.newaxis]) * (1 - white[:, :, np.newaxis]) + white[:, :, np.newaxis] * 255, np.uint8))

    cv2.waitKey(1)
