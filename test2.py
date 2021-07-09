import asyncio
import cv2
import numpy as np
import websockets
import base64
import json
import os
import imutils
import time
# custom import
from my_matcher import *
from utils import *

CORONA_FILENAMES = ("eye-character-1.png", "eye-character-2.png", "eye-character-3.png", "eye-character-4.png") 
CHAR_DIR = "characters"
DOCTOR_FILENAMES = ["character-6.png", "left-character-6.png", "left-left-character-6.png", "right-character-6.png", "right-right-character-6.png"]
DOCTOR_ORG_FILENAME = ["character-6.png"]
corona_templates = get_list(CHAR_DIR, CORONA_FILENAMES)
doctor_templates = get_list(CHAR_DIR, DOCTOR_FILENAMES)
#used_id = set()

def get_bounds(templates, wave, threshold = 0.8, multiscale = False, n_cut = 1):
    bounds = []
    for template in templates:
        if not multiscale:
            template = imutils.resize(template, width = int(template.shape[1] * 0.5))
            new_bounds = template_matcher(template, wave, threshold=threshold)
        else:
            new_bounds = template_matcher_multiscale(template, wave, threshold = threshold, n_level = 3, scale_range = [1.7, 2.5])
        #new_bounds = remove_overlap(new_bounds)
        if n_cut and len(new_bounds) > n_cut:
            bounds.extend(new_bounds[:n_cut])
        else:
            bounds.extend(new_bounds)
    # remove bound that is overlapped
    bounds = remove_overlap(bounds)
    return bounds

def catch_corona2(wave):
    # detect each corona's and doctor's rectangle bound
    corona_bounds = get_bounds(CORONA_FILENAMES, wave, threshold = 0.8)
    doctor_bounds = get_bounds(doctor_templates, wave, threshold = 0.3, n_cut=5)
    
    # calculate result, choose one point for each rectangle
    results = []
    for top_left, bottom_right in corona_bounds:
        x = (top_left[0] + bottom_right[0]) // 2
        y = (top_left[1] + bottom_right[1]) // 2
        if not is_in_doctor_bound(doctor_bounds, (x, y)):
            results.append((x, y))

    return results

def catch_corona(wave):
    # detect each corona's and doctor's rectangle bound
    corona_bounds = get_bounds(corona_templates, wave, threshold = 0.8, n_cut=None)
    doctor_points = SIFT_detector_FLANN_matching(doctor_templates[0], wave)
    doctor_bounds = get_bounds(doctor_templates[1:], wave, threshold = 0.3, n_cut=5)
    
    # calculate result, choose one point for each rectangle
    # remove all point in doctor's zone
    r = 50
    results = []
    for top_left, bottom_right in corona_bounds:
        x = (top_left[0] + bottom_right[0]) // 2
        y = (top_left[1] + bottom_right[1]) // 2
        if not is_in_doctor_point(doctor_points, (x, y), r)\
            and not is_in_doctor_bound(doctor_bounds, (x, y)):
            results.append((x, y))

    return results

def solve(json_datas):
    for i, json_data in enumerate(json_datas):
        wave = base64_to_image(json_data['base64Image'])
        wave_id = json_data["waveId"]
        round_id = json_data["roundId"]

        ### catch corona in a wave image
        results = catch_corona(wave)
        print(i)
        # print(results)
        draw_circle(wave, results, round_id, wave_id)

with open("test1.json", "r") as f:
    json_datas = json.load(f)
solve(json_datas)
