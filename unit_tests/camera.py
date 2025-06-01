"""
If logged in remotely, please enable X11 forwarding, either `ssh -X` or `ssh -Y`
"""

import sys
import os
import json
import cv2 as cv
from picamera2 import Picamera2
from time import time

print("Please adjust lens focus if image is blurry")
# SETUP
# Load configs
params_file_path = os.path.join(os.path.dirname(sys.path[0]), "configs.json")
with open(params_file_path, "r") as file:
    params = json.load(file)
# Config Pi Camera
cv.startWindowThread()
cam = Picamera2()
cam.configure(
    cam.create_preview_configuration(
        main={"format": "RGB888", "size": (224, 224)},
        controls={
            "FrameDurationLimits": (
                int(1_000_000 / params["frame_rate"]),
                int(1_000_000 / params["frame_rate"]),
            )
        },
    )
)
# Start Pi Camera with a count down
cam.start()
for i in reversed(range(3 * params["frame_rate"])):
    frame = cam.capture_array()
    if frame is None:
        print("No frame received. TERMINATE!")
        sys.exit()
    if not i % params["frame_rate"]:
        print(i / params["frame_rate"])  # count down 3, 2, 1 sec
# Init timer for FPS computing
frame_counts = 0
frame_rate = 0.0
start_stamp = time()

# LOOP
try:
    while True:
        if frame is None:
            print("No frame received. TERMINATE!")
            break
        im = cam.capture_array()
        # Log frame rate
        frame_counts += 1
        since_start = time() - start_stamp
        frame_rate = frame_counts / since_start
        print(f"frame rate: {frame_rate}")
        # grey = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
        cv.imshow("Camera", im)
        if cv.waitKey(1) == ord("q"):  # [q]uit
            print("Quit signal received.")
            cv.destroyAllWindows()
            sys.exit()
except KeyboardInterrupt:
    cv.destroyAllWindows()
    sys.exit()
finally:
    cv.destroyAllWindows()
    sys.exit()
