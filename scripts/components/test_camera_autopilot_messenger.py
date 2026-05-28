from pathlib import Path
import json
from time import sleep, time
from camera import ThreadedCamera
from messenger import ThreadedMessenger
from driver import Driver
import cv2 as cv

# SAFETY CHECK
is_lifted = input("Is anything contacting any wheels of BearCar? (Y/n)")
while is_lifted != "n":
    print("Please lift BearCar up and remove everything that is making the contact")
    is_lifted = input("Is anything contacting any wheels of BearCar? (Y/n)")
print("Hold tight! You are about to unleash the beast!")
print("Verify light order: cyan -> yellow -> green -> blue -> purple -> red")

# SETUP
# Load configs
params_file_path = Path(__file__).parent.joinpath("configs.json")
with open(params_file_path, "r") as file:
    params = json.load(file)
# Init components
picam = ThreadedCamera()
messenger = ThreadedMessenger(port="/dev/ttyACM0", baudrate=115200)
driver = Driver(joy_id=0, autopilot_name="example_pilot")

# LOOP
try:
    frame_counts = 0
    start_time = time()
    while not driver.is_terminated:
        frame = picam.read()
        frame_counts += 1
        mode, st_pw, th_pw = driver.process_event(params, frame=frame)
        messenger.out_msg = f"{mode},{st_pw},{th_pw}\n"
        if not frame_counts % params["frame_rate"]:
            elapsed = time() - start_time
            fps = params["frame_rate"] / elapsed
            print("---")
            print("Out message: " + messenger.out_msg)
            print(f"angular velocity on z: {messenger.ang_vel_z}")
            print(f"Processing at {fps:.2f} FPS | Frame shape: {frame.shape}")
            start_time = time()
        cv.imshow("Camera", cv.flip(frame, -1))  # picam mounted upside down
        if cv.waitKey(1) == ord("q"):  # [q]uit
            print("Quit signal received.")
            break
        sleep(1 / params["frame_rate"])  # see configs.json for FPS
except KeyboardInterrupt:
    print("\nShutdown signal received.")
finally:
    picam.stop()
