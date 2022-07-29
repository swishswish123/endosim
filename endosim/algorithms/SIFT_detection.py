import cv2
import matplotlib.pyplot as plt
import os
import numpy as np


def SIFT_matching(previous_image, current_image):

    # read images
    img1 = previous_image
    img2 = current_image

    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    #figure, ax = plt.subplots(1, 2, figsize=(16, 8))
    #ax[0].imshow(img1, cmap='gray')
    #ax[1].imshow(img2, cmap='gray')
    #plt.show()

    # GENERATE SIFT FEATURES
    sift = cv2.xfeatures2d.SIFT_create()

    keypoints_1, descriptors_1 = sift.detectAndCompute(img1,None)
    keypoints_2, descriptors_2 = sift.detectAndCompute(img2,None)

    print(len(keypoints_1), len(keypoints_2))
    print('fuond key points')
    # feature matching
    #bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descriptors_1,descriptors_2,k=2)
    #matches = bf.match(descriptors_1,descriptors_2)
    print('foound matches')

    # Apply ratio test
    good = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good.append([m])

    #best_sorted_matches = sorted(good, key = lambda x:x.distance)
    print('sorted matches')

    print(len(matches))
    #print(len(best_sorted_matches))
    #img3 = cv2.drawMatches(img1, keypoints_1, img2, keypoints_2, best_sorted_matches, img2, flags=2)
    img3 = cv2.drawMatchesKnn(img1,keypoints_1,img2,keypoints_2,good,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    print('draw matches')

        #,plt.show()

    return img3


def remove_speckle(gray):

    # first we want to convert it to float (so that we can make zeros nan)
    gray = gray.astype('float')

    mean = gray.mean()
    std = gray.std()

    fig, ax = plt.subplots(1, 2)
    # original
    ax[0].imshow(gray)
    ax[0].set_title('original image')
    # speckle
    speckle = gray>mean+3*std
    ax[1].set_title('speckle')
    ax[1].imshow(speckle)
    plt.show()

    return


def read_video():

    cap = cv2.VideoCapture(os.path.join('videos','051_01_trimmed.MP4'))
    frame_num = 0
    previous_image = np.array([])
    current_image = np.array([])
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        # if frame is read correctly ret is True
        if not ret:
            print("something went wrong...")
            break
        # turn to gray
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # remove speckle
        #remove_speckle(gray)

        if frame_num==0:
            print(f'frame {frame_num}')
            current_image = frame
        else:
            print(f'frame  {frame_num}')
            print(current_image.shape, previous_image.shape)
            previous_image = current_image
            current_image = frame

            img3 = SIFT_matching(previous_image, current_image)

            cv2.imshow('frame',img3)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        frame_num += 1
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


def main():
    read_video()
    return


if __name__ == '__main__':
    main()