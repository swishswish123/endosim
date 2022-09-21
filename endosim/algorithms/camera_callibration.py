

import numpy as np
import cv2
import glob
import keyboard

def find_corners(cap,criteria,pattern_size = (14, 10)):

    ret, img = cap.read()
    cv2.imshow('frame', img)

    # turn to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        #objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        #imgpoints.append(corners)
        # Draw and display the corners
        cv2.drawChessboardCorners(img, pattern_size, corners2, ret)
        cv2.imshow('img', img)
        cv2.waitKey(500)


def main():
    # start video capture
    cap = cv2.VideoCapture(0)

    # Check if the webcam is opened correctly
    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    # number of inner corners in chessboard in rows and columns
    pattern_size = (14, 10)
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6*7,3), np.float32)
    objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    while True:
        #find_corners(cap, criteria)
        # read and display image from capture
        ret, img = cap.read()
        cv2.imshow('frame', img)

        # turn frame to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)


        # If found, and we want to capture add object points, image points (after refining them)
        if ret == True:

            corners2 = cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
            # Draw and display the corners
            cv2.drawChessboardCorners(img, pattern_size, corners2, ret)
            cv2.imshow('img', img)

            if cv2.waitKey(20)==32:

                objpoints.append(objp)
                imgpoints.append(corners)
                print(len(objpoints))
                print(len(imgpoints))
                cv2.waitKey(500)

            cv2.waitKey(500)

        c = cv2.waitKey(1)
        if c == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()