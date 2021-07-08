import json
import base64
import numpy as np
import cv2

def make_json_round(id_round, catchings):
    return json.dumps({"roundId": id_round, "catchings": catchings})

def make_obj(id_wave, x, y):
    return {"waveID": id_wave, "position": [{"x": x, "y": y}]}

def base64_to_image(base64_data):
    encoded_data = base64_data.split(',')[1]
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)

    return img

if __name__ == '__main__':
    catchings = [make_obj("123", 15, 5), make_obj("3234", 21, 214)]
    print(make_json_round("abc", catchings))