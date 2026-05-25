import sys
from pathlib import Path
import json
from time import sleep
from messenger import Messenger
from drive_manager import DriveManager


# SETUP
# Load configs
params_file_path = str(Path(__file__).parents[1].joinpath("configs.json"))
with open(params_file_path, "r") as file:
    params = json.load(file)
# Init serial port
messenger = Messenger(port="/dev/ttyACM0", baudrate=115200)
manager = DriveManager(joy_id=0, autopilot_on=False)

# MAIN LOOP
try:
    while not manager.is_terminated:
        mode, st_pw, th_pw = manager.process_events(params)
        messenger.out_msg = f"{mode},{st_pw},{th_pw}\n"
        print("---")
        print("Out message: " + messenger.out_msg)
        print(f"angular velocity on z: {messenger.ang_vel_z}")
        # 50Hz
        sleep(0.02)
except KeyboardInterrupt:
    sys.exit()
