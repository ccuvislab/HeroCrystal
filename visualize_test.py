import json
from detectron2.utils.file_io import PathManager


inputfile = 'temp.json'
with PathManager.open(inputfile, "r") as f:
        predictions = json.load(f)
