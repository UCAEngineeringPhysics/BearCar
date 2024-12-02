"""
If logged in remotely, please enable X11 forwarding, either `ssh -X` or `ssh -Y`
"""
import sys
import os
import cv2
from picamera2 import Picamera2
from time import sleep

print("Please adjust lens focus if image is blurry")
# SETUP
# Config Pi Camera
for i in reversed(range(1, 4)):
    print(i)
    sleep(1)
cv2.startWindowThread()
picam = Picamera2()
picam.configure(
    picam.create_preview_configuration(
        main={"format": 'RGB888', "size": (224, 224)},
        controls={"FrameDurationLimits": (41667, 41667)},  # 24 FPS
    )
)
picam2.start()

# LOOP
while True:
    im = picam.capture_array()
    grey = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Camera", im)
    # Press "q" to quit
    if cv2.waitKey(1)==ord('q'):
        cv2.destroyAllWindows()
        sys.exit()
