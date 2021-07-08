import cv2
import argparse
from my_matcher import *

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-t", "--template", required=True, help="Path to template image")
ap.add_argument("-i", "--image", required=True,
	help="Path to images where template will be matched")
args = vars(ap.parse_args())

# load the template image, convert it to grayscale
template = cv2.imread(args["template"], cv2.IMREAD_UNCHANGED)
template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
template = imutils.resize(template, width = int(template.shape[1] * 0.5))

# load wave image, convert it to gray scale
wave = cv2.imread(args["image"], cv2.IMREAD_UNCHANGED)
gray_wave = cv2.cvtColor(wave, cv2.COLOR_BGR2GRAY)

# template matching
template_matcher(template, gray_wave, threshold = 0.8, visualize = True)
