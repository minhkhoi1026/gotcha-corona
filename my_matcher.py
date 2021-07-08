# import the necessary packages
from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import argparse
import imutils
import glob
import cv2


def template_matcher(template_image, wave_image, method = cv2.TM_CCOEFF_NORMED, threshold = 0.8):
    template_image = cv2.cvtColor(template_image, cv2.COLOR_BGR2GRAY)
    wave_image = cv2.cvtColor(wave_image, cv2.COLOR_BGR2GRAY)
    # loop over the images to find the template in
    matches = cv2.matchTemplate(wave_image, template_image, method)

    # Store the coordinates of matched area in a numpy array
    w, h = template_image.shape[::-1]
    # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
    if method == cv2.TM_SQDIFF or method == cv2.TM_SQDIFF_NORMED:
        locs = np.where(matches <= threshold)
    else: 
        locs = np.where(matches >= threshold)
    
    results = []
    for loc in zip(*locs[::-1]):
        top_left = loc
        bottom_right = (top_left[0] + w, top_left[1] + h)
        results.append([top_left, bottom_right])

    return results


def ORB_dectector_BF_matching(template_image, gray_wave_image):
    # ORB Detector
    orb = cv2.ORB_create()
    #orb = cv2.SIFT.create()
    kp1, des1 = orb.detectAndCompute(template_image, None)
    kp2, des2 = orb.detectAndCompute(gray_wave_image, None)
    # Brute Force Matching
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key = lambda x:x.distance)
    #drawing the matches
    matching_result = cv2.drawMatches(template_image, kp1, gray_wave_image, kp2, matches[:50], None, flags=2)
    cv2.imshow('Good Matches', matching_result)
    cv2.waitKey()

def SIFT_detector_FLANN_matching(template_image, gray_wave_image, threshold = 0.7):
    # Initiate SIFT detector
    sift = cv2.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(template_image,None)
    kp2, des2 = sift.detectAndCompute(gray_wave_image,None)

    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)   # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k = 2)

    # Need to draw only good matches, so create a mask
    matchesMask = [[0,0] for i in range(len(matches))]

    # ratio test as per Lowe's paper
    for i, (m, n) in enumerate(matches):
        if m.distance < threshold * n.distance:
            matchesMask[i] = [1,0]
    draw_params = dict(matchColor = (0, 255, 0),
                    singlePointColor = (255, 0, 0),
                    matchesMask = matchesMask,
                    flags = cv2.DrawMatchesFlags_DEFAULT)
    img3 = cv2.drawMatchesKnn(template_image, kp1, gray_wave_image, kp2, matches, None, **draw_params)
    plt.imshow(img3,),plt.show()