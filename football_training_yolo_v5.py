

from roboflow import Roboflow
rf = Roboflow(api_key="kR3QPcEObgoD4Qzn8tKs")
project = rf.workspace("roboflow-jvuqo").project("football-players-detection-3zvbc")
version = project.version(1)
dataset = version.download("yolov5")

import shutil
shutil.move('football-players-detection-1/train',
            'football-players-detection-1/train'
            )

shutil.move('football-players-detection-1/test',
            'football-players-detection-1/test'
            )

shutil.move('football-players-detection-1/valid',
            'football-players-detection-1/valid'
            )




