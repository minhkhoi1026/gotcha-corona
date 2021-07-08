import asyncio
import cv2
import numpy as np
import websockets
import base64
import json
import os
import imutils
# custom import
from my_matcher import *
from utils import *

EYE_CORONA_PATHS = ("eye-character-1.png", "eye-character-2.png", "eye-character-3.png", "eye-character-4.png") 
CHAR_DIR = "characters"

def catch_corona(wave_image, wave_id):
    corona_bounds = []
    for path in EYE_CORONA_PATHS:
        full_path = os.path.join(CHAR_DIR, path)
        template_image = cv2.imread(full_path, cv2.IMREAD_UNCHANGED)
        template_image = imutils.resize(template_image, width = int(template_image.shape[1] * 0.5))
        corona_bounds.extend(template_matcher(template_image, wave_image))
    corona_bounds = remove_overlap(corona_bounds)
    
    results = []
    for top_left, bottom_right in corona_bounds:
        x = (top_left[0] + bottom_right[0]) // 2
        y = (top_left[1] + bottom_right[1]) // 2
        results.append((x, y))

    return results

async def play_game(websocket, path):
    print('Corona Killer is ready to play!')
    catchings = []
    last_round_id = ''
    wave_count = 0
    
    while True:

        ### receive a socket message (wave)
        try:
            data = await websocket.recv()
        except Exception as e:
            print('Error: ' + e)
            break

        json_data = json.loads(data)

        ### check if starting a new round
        if json_data["roundId"] != last_round_id:
            #print(f'> Catching corona for round {json_data["roundId"]}...')
            last_round_id = json_data["roundId"]

        ### catch corona in a wave image
        wave_image = base64_to_image(json_data['base64Image'])
        wave_id = json_data["waveId"]
        results = catch_corona(wave_image, wave_id)

        ### store catching positions in the list
        catchings.append(make_wave_dict(wave_id, results))

        print(f'>>> Wave #{wave_count:03d}: {json_data["waveId"]}')
        wave_count = wave_count + 1

        ### send result to websocket if it is the last wave
        if json_data["isLastWave"]:
            round_id = json_data["roundId"]
            print(f'> Submitting result for round {round_id}...')

            json_result = make_round_json(round_id, catchings)
            with open("test.json", "w") as f:
                f.write(json_result)

            await websocket.send(json_result)
            print('> Submitted.')

            catchings = []
            wave_count = 0


start_server = websockets.serve(play_game, "localhost", 8765, max_size=100000000)

asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()
