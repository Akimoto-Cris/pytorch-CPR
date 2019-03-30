import argparse
import json
import os

def load_config(json_path):
    with open(json_path, 'rb') as f:
        config = json.loads(f.read())
        config["env"]["saveDir"] = os.path.join(config["env"]["expDir"], config["env"]["expID"])
        if not os.path.exists(config["env"]["saveDir"]):
            os.makedirs(config["env"]["saveDir"])
        return config

class Opts:
    def __init__(self):
        self.parser = argparse.ArgumentParser()

    def init(self):
        self.parser.add_argument("--config_path", "-c", default=None, type=str, help="path/to/config_json/file.")

    def parse(self):
        self.init()
        self.opt = self.parser.parse_args()
        config = load_config(self.opt.config_path)
        return config