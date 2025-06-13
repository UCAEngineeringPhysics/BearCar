import sys
import os
import json
from time import time
import pygame
import torch
import torch.nn as nn
from torchvision.transforms import v2
import cv2 as cv
from picamera2 import Picamera2


# SETUP
# Define BearNet
class BearNet(nn.Module):
    def __init__(self):
        super(BearNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=2)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3)
        self.conv7 = nn.Conv2d(256, 256, kernel_size=3)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # self.avg_pool = nn.AvgPool2d(kernel_size=1)
        self.fc1 = nn.Linear(256 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 2)

    def forward(self, x):  # 224
        x = self.relu(self.conv1(x))  # (224 - 7 + 2 * 3) / 2 + 1 = 112.5
        x = self.max_pool(x)  # (112 - 3 + 2 * 1) / 2 + 1 = 56.5
        x = self.relu(self.conv2(x))  # (56 - 3) + 1 = 54
        x = self.relu(self.conv3(x))  # (54 - 3) / 2 + 1 = 26.5
        x = self.relu(self.conv4(x))  # (26 - 3) + 1 = 24
        x = self.relu(self.conv5(x))  # (24 - 3) / 2 + 1 = 11.5
        x = self.relu(self.conv6(x))  # (11 - 3) + 1 = 9
        x = self.relu(self.conv7(x))  # (9 - 3) + 1 = 7
        # x = self.avg_pool(x)  # (7 - 1) + 1 = 7
        x = x.view(x.size(0), -1)  # flatten
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        y = self.fc3(x)
        return y


# Instantiate BearNet
random_pilot = BearNet()
random_pilot.eval()
# Load configs
params_file_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs.json"
)
with open(params_file_path, "r") as file:
    params = json.load(file)
# Config image transforms
to_tensor = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
# Init controller
pygame.display.init()
pygame.joystick.init()
js = pygame.joystick.Joystick(0)
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
# Variables
is_paused = True


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
        for e in pygame.event.get():  # read controller input
            if e.type == pygame.JOYBUTTONDOWN:
                if js.get_button(params["record_btn"]):
                    is_paused = not is_paused
                    print(f"Paused: {is_paused}")
                elif js.get_button(params["stop_btn"]):  # emergency stop
                    print("E-STOP PRESSED. TERMINATE!")
                    cv.destroyAllWindows()
                    pygame.quit()
                    sys.exit()
        # Predict steer and throttle
        img_tensor = to_tensor(frame)
        with torch.no_grad():
            pred_st, pred_th = map(
                float,
                torch.clamp(
                    random_pilot(img_tensor[None, :]).squeeze(), min=-0.999, max=0.999
                ),
            )
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
