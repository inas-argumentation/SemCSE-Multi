import json
import os

def relative_path(path):
    return os.path.join(os.path.dirname(__file__), path)

def load_invasion_dataset():
    with open(relative_path("../data/invasion_dataset.json"), "r") as f:
        dataset = json.load(f)
    return dataset