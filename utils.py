import json
import base64
import numpy as np
import cv2
import os

def make_round_json(round_id, catchings):
    return json.dumps({"roundId": round_id, "catchings": catchings})

def make_wave_dict(wave_id, result):
    positions = []
    for x, y in result:
        positions.append({"x": int(x), "y": int(y)})
    return {"waveID": wave_id, "positions": positions}

def base64_to_image(base64_data):
    encoded_data = base64_data.split(',')[1]
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
    return img

def remove_overlap(bounds, overlap_threshold = 0.25):
    results = []
    # determine if current match overlapped with other match
    # Only add match which not overlapped with any match
    for i, (top_left, bottom_right) in enumerate(bounds):
        overlapped = False
        for j, (pre_top_left, pre_bottom_right) in enumerate(bounds):
            if (j <= i): continue
            # calculate overlapping area
            w_overlap = min(pre_bottom_right[0], bottom_right[0]) - max(pre_top_left[0], top_left[0])
            h_overlap = min(pre_bottom_right[1], bottom_right[1]) - max(pre_top_left[1], top_left[1])
            w = (pre_bottom_right[1] - pre_top_left[1]) 
            h = (pre_bottom_right[0] - pre_top_left[0]) 
            # two match are overlap if they share at least @overlap_threshold percent area 
            if w_overlap > 0 and h_overlap > 0 and w_overlap * h_overlap >= overlap_threshold * w * h:
                overlapped = True
                break
        if not overlapped:
            results.append([top_left, bottom_right])
    return results

# NOT DONE YET
def draw_circle(wave_image, points, round_id, wave_id):
    ### save result image file for debugging purpose
    for point in points:
        cv2.circle(wave_image, point, radius=10, color=(0, 0, 255), thickness=-1)
    waves_dir = f'waves/{round_id}/'
    if not os.path.exists(waves_dir):
        os.makedirs(waves_dir)
    cv2.imwrite(os.path.join(waves_dir, f'{wave_id}.jpg'), wave_image)

def is_in_doctor_bound(doctor_bounds, point):
    for top_left, bottom_right in doctor_bounds:
        x = (top_left[0] + bottom_right[0]) // 2
        y = (top_left[1] + bottom_right[1]) // 2
        if (top_left[0] - 30 <= point[0] and point[0] <= bottom_right[0] + 30)\
            and (top_left[1] - 30 <= point[1] and point[1] <= bottom_right[1] + 30):
            return True
        # if (x - 30 <= point[0] and point[0] <= x + 30)\
        #      and (y - 30 <= point[1] and point[1] <= y + 30):
        #      return True
    return False

def is_in_doctor_point(doctor_points, point, r):
    for x, y in doctor_points:
        if (x - r <= point[0] and point[0] <= x + r)\
            and (y - r <= point[1] and point[1] <= y + r):
            return True
    return False

def get_list(dir, filenames):
    images = []
    for filename in filenames:
        full_path = os.path.join(dir, filename)
        images.append(cv2.imread(full_path, cv2.IMREAD_UNCHANGED))
    return images