# coding=utf-8
""" Module for multiplying numbers. """
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt


def plt_image_histogram(gray):
    plt.figure()
    plt.hist(gray.ravel(),256,[1,255])
    plt.title('histogram')
    plt.show()

    return


def plt_colored_hist(frame):
    color = ('b','g','r')
    for i,col in enumerate(color):
        histr = cv2.calcHist([frame],[i],None,[256],[1,256])
        plt.plot(histr,color = col)
        plt.xlim([0,256])
    plt.show()


def display_frame(frame):
    # Display the resulting frame
    cv2.imshow('frame', frame)


def plt_points(gray, frame):
    # first we want to convert it to float (so that we can make zeros nan)
    gray = gray.astype('float')
    gray[gray==0]=np.nan

    mean = np.nanmean(gray)
    std = np.nanstd(gray)

    fig, ax = plt.subplots(2, 2)
    # original
    ax[0,0].imshow(frame)
    ax[0,0].set_title('original image')
    # speckle
    speckle = gray>mean+3*std
    ax[0,1].set_title('speckle')
    ax[0,1].imshow(speckle)
    # within 1 std
    mask = (gray >= mean - std) * (gray<=mean+std)
    ax[1,0].imshow(mask)
    ax[1,0].set_title('within 1std')

    # between 1and2 std
    mask_lower = (gray >= mean - 2*std) * (gray<=mean-std)
    mask_upper = (gray >= mean + std) * (gray<=mean+2*std)
    mask_total = mask_lower+mask_upper
    ax[1,1].imshow(mask_total)
    ax[1,1].set_title('between 1 and 2 std')
    plt.show()

    return


def get_corrected_img(img1, img2):
    MIN_MATCHES = 50

    orb = cv2.ORB_create(nfeatures=500)
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    index_params = dict(algorithm=6,
                        table_number=6,
                        key_size=12,
                        multi_probe_level=2)
    search_params = {}
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # As per Lowe's ratio test to filter good matches
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    if len(good_matches) > MIN_MATCHES:
        src_points = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_points = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        m, mask = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)
        corrected_img = cv2.warpPerspective(img1, m, (img2.shape[1], img2.shape[0]))

        return corrected_img
    return img2


def detect_points():

    cap = cv2.VideoCapture(os.path.join('videos','051_01_trimmed.MP4'))

    frame_num = 0

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        # turn to gray
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if frame_num == 0 or frame_num==10 or frame_num==300:
            #plt_image_histogram(gray)
            #plt_colored_hist(frame)


            plt_points(gray, frame)
            if frame_num == 0:
                im1 = gray
            if frame_num == 300:
                im2 = gray
        #speckle = np.where(gray >200)
        #plt.imshow(frame)
        #plt.scatter(speckle[0], speckle[1],marker='x', color="white")
        # display_frame(frame)

        if cv2.waitKey(1) == ord('q'):
            break
        frame_num +=1


    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

    return im1, im2


def main():
    im1, im2= detect_points()
    img = get_corrected_img(im1, im2)
    cv2.imshow('Corrected image', img)
    cv2.waitKey()
    return

if __name__ == '__main__':
    main()
