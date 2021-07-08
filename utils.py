import json
import base64
import numpy as np
import cv2

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
def draw_bound(wave_image, bounds):
    cv2.circle(wave_image, (x,y), radius=10, color=(0, 0, 255), thickness=-1)

    ### save result image file for debugging purpose
    for result in bounds:
        cv2.rectangle(wave_image, result[0], result[1], (0, 0, 255), 2)
    waves_dir = f'waves/abc/'
    if not os.path.exists(waves_dir):
        os.makedirs(waves_dir)
    cv2.imwrite(os.path.join(waves_dir, f'{wave_id}.jpg'), wave_image)