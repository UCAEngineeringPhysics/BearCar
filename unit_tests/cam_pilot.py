import sys
import os
import json
from time import time
import torch
import torch.nn as nn
from torchvision.transforms import v2
import cv2 as cv
from picamera2 import Picamera2
from ..bc_models.bearnet import BearNet


# SETUP
# Instantiate BearNet
random_pilot = BearNet()
random_pilot.eval()
# Load configs
params_file_path = os.path.join(os.path.dirname(sys.path[0]), "configs.json")
with open(params_file_path, "r") as file:
    params = json.load(file)
# Config image transforms
to_tensor = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
# Init camera
cv.startWindowThread()
cam = Picamera2()
cam.configure(
    cam.create_preview_configuration(
        main={"format": "RGB888", "size": (224, 224)},
        controls={
            "FrameDurationLimits": (
                int(1000_000 / params["frame_rate"]),
                int(1000_000 / params["frame_rate"]),
            )
        },
    )
)
cam.start()
for i in reversed(range(3 * params["frame_rate"])):
    frame = cam.capture_array()
    if frame is None:
        print("No frame received. TERMINATE!")
        sys.exit()
    if not i % params["frame_rate"]:
        print(i / params["frame_rate"])  # count down 3, 2, 1 sec
# Init timer for FPS computing
start_stamp = time()
frame_counts = 0
ave_frame_rate = 0.0


# LOOP
try:
    while True:
        frame = cam.capture_array()  # read image
        if frame is None:
            print("No frame received. TERMINATE!")
            break
        cv.imshow("Camera", frame)
        # Log frame rate
        frame_counts += 1
        since_start = time() - start_stamp
        frame_rate = frame_counts / since_start
        print(f"frame rate: {frame_rate}")  # debug
        # Predict steer and throttle
        img_tensor = to_tensor(frame)
        with torch.no_grad():
            pred_st, pred_th = map(float, random_pilot(img_tensor[None, :]).squeeze())
        print(pred_st, pred_th)
        if cv.waitKey(1) == ord("q"):
            print("Quit signal received.")
            break
except KeyboardInterrupt:  # take care terminate signal (Ctrl-c)
    cv.destroyAllWindows()
    sys.exit()
finally:
    cv.destroyAllWindows()
    sys.exit()
