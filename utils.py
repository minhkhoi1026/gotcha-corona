import json

def make_json_round(id_round, catchings):
    return json.dumps({"roundId": id_round, "catchings": catchings})

def make_obj(id_wave, x, y):
    return {"waveID": id_wave, "position": [{"x": x, "y": y}]}

catchings = [make_obj("123", 15, 5), make_obj("3234", 21, 214)]
print(make_json_round("abc", catchings))