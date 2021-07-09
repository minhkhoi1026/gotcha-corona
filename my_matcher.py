# import the necessary packages
from __future__ import print_function
from utils import remove_overlap
import matplotlib.pyplot as plt
import numpy as np
import argparse
import imutils
import glob
import cv2
import copy

'''template matching using tradiontinal template matching algorithm'''
def template_matcher(template, wave, method = cv2.TM_CCOEFF_NORMED, threshold = 0.8):
    template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    wave = cv2.cvtColor(wave, cv2.COLOR_BGR2GRAY)
    # loop over the images to find the template in
    matches = cv2.matchTemplate(wave, template, method)

    # Store the coordinates of matched area in a numpy array
    w, h = template.shape[::-1]
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
    cmp = lambda result: -matches[result[0][1], result[0][0]]
    results.sort(key = cmp)
    return results

'''template matching using template matching algorithm 
combine with image pyramid for multiscale matching'''
def template_matcher_multiscale(template, wave, threshold = 0.8, scales = [0.5, 0.3]):
    # Convert to grayscale
    # image =  copy.deepcopy(wave)
    wave = cv2.cvtColor(wave, cv2.COLOR_BGR2GRAY)
    template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    
    # Store width and height of template in w and h
    w, h = template.shape[::-1]
    found = None
    results = []
    temps = []
    for scale in scales:
        # resize the image according to the scale, and keep track
        # of the ratio of the resizing
        resized_template = imutils.resize(template, width = int(template.shape[1] * scale))

        matches = cv2.matchTemplate(wave, resized_template, cv2.TM_CCOEFF_NORMED)
        # get location of all matches
        locs = np.where(matches >= threshold)
        
        for loc in zip(*locs[::-1]):
            temps.append([matches[loc[1], loc[0]], loc, scale])
  
    cmp = lambda item: -item[0]
    temps.sort(key = cmp)
    results = []
    for _, top_left, r in temps:
        bottom_right = (top_left[0] + w * r, top_left[1] + h * r)
        results.append([top_left, bottom_right])

    return results

'''Feature matching using ORB detector and Brute Force matching'''
def ORB_detector_BF_matching(template, wave):
    # ORB Detector
    orb = cv2.ORB_create()
    #orb = cv2.SIFT.create()
    kp1, des1 = orb.detectAndCompute(template, None)
    kp2, des2 = orb.detectAndCompute(wave, None)
    # Brute Force Matching
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key = lambda x:x.distance)
    #drawing the matches
    matching_result = cv2.drawMatches(template, kp1, wave, kp2, matches[:50], None, flags=2)
    cv2.imshow('Good Matches', matching_result)
    cv2.waitKey()

'''Template matching using SIFT detector and FLANN matching'''
def SIFT_detector_FLANN_matching(template, wave, threshold = 0.75, visualize = False):
    # Initiate SIFT detector
    sift = cv2.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(template,None)
    kp2, des2 = sift.detectAndCompute(wave,None)

    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)   # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k = 2)

    # Need to draw only good matches, so create a mask
    matchesMask = [[0,0] for i in range(len(matches))]

    # ratio test as per Lowe's paper
    results = []
    for i, (m, n) in enumerate(matches):
        if m.distance < threshold * n.distance:
            matchesMask[i] = [1,0]
            results.append(kp2[m.trainIdx].pt)

    if visualize:
        draw_params = dict(matchColor = (0, 255, 0),
                        singlePointColor = (255, 0, 0),
                        matchesMask = matchesMask,
                        flags = cv2.DrawMatchesFlags_DEFAULT)
        img3 = cv2.drawMatchesKnn(template, kp1, wave, kp2, matches, None, **draw_params)
        cv2.imshow('Good Matches', img3)
        cv2.waitKey()

    return results