from pathlib import Path
import json
from time import sleep, time
from components.camera import ThreadedCamera
from components.messenger import ThreadedMessenger
from components.driver import Driver
import cv2 as cv
from datetime import datetime
import csv

# SAFETY CHECK
print("Place BearCar on the ground and enjoy your ride!")

# SETUP
# Define paths
bc_dir = Path(__file__).parents[1]
data_dir = bc_dir.joinpath("data")
image_dir = str(data_dir.joinpath(datetime.now().strftime("%Y-%m-%d-%H-%M"), "images"))
Path(image_dir).mkdir(parents=True, exist_ok=True)
label_path = str(Path(image_dir).parent.joinpath("labels.csv"))
# Load configs
params_file_path = bc_dir.joinpath("scripts", "components", "configs.json")
with open(params_file_path, "r") as file:
    params = json.load(file)
# Init components
picam = ThreadedCamera()
messenger = ThreadedMessenger(port="/dev/ttyACM0", baudrate=115200)
driver = Driver(joy_id=0, autopilot_model=None)

# LOOP
try:
    frame_counts = 0
    record_counts = 0
    start_time = time()
    while not driver.is_terminated:
        frame = picam.read()
        frame_counts += 1
        mode, st_pw, th_pw = driver.process_event(params, frame=None)
        messenger.out_msg = f"{mode},{st_pw},{th_pw}\n"
        action = [st_pw, th_pw]
        if mode == "r":
            cv.imwrite(image_dir + "/" + str(frame_counts) + ".jpg", frame)
            label = [str(frame_counts) + ".jpg"] + action
            with open(label_path, "a+", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(label)
            record_counts += 1
            print(f"recorded frame counts: {record_counts}")  # debug
            if record_counts == params["record_cap"]:  # pause recording if max reached
                driver.mode = "p"
                driver.is_paused = True
                driver.is_recording = False
        if not frame_counts % params["frame_rate"]:
            elapsed = time() - start_time
            fps = params["frame_rate"] / elapsed
            print("---")
            print("Out message: " + messenger.out_msg)
            print(f"angular velocity on z: {messenger.ang_vel_z}")
            print(f"Processing at {fps:.2f} FPS | Frame shape: {frame.shape}")
            start_time = time()
        # cv.imshow("Camera", cv.flip(frame, -1))  # picam mounted upside down
        # if cv.waitKey(1) == ord("q"):  # [q]uit
        #     print("Quit signal received.")
        #     break
        sleep(1 / params["frame_rate"])  # see configs.json for FPS
except KeyboardInterrupt:
    print("\nShutdown signal received.")
finally:
    picam.stop()
