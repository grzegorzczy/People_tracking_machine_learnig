import numpy as np
import os
import random
import shutil
from ultralytics import YOLO

model = YOLO("yolov8n-seg.pt")

#train the model
results = model.train(data="data.yaml", name="detector_model", epochs=15, imgsz=640, exist_ok=True)