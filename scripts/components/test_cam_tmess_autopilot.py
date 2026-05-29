from pathlib import Path
import json
from time import sleep, time
from picamera2 import Picamera2
import cv2 as cv
import torch

from messenger import ThreadedMessenger
from driver import Driver
from autopilot_architectures.bearnet import BearNet

# SAFETY CHECK
is_tangled = input("Observe BearCar closely! Is any wire touching the wheels? (Y/n)")
while is_tangled != "n":
    print("Please decouple the wires from BearCar's wheels!")
    is_tangled = input(
        "Observe BearCar closely! Is any wire touching the wheels? (Y/n)"
    )
print("!!!\nAUTOPILOT ON DUTY\n!!!")


# SETUP
# Load configs
params_file_path = Path(__file__).parent.joinpath("configs.json")
with open(params_file_path, "r") as file:
    params = json.load(file)
# Load autopilot model
model = BearNet()
model_path = Path(__file__).parents[2].joinpath("models", "dummy_pilot")
model.load_state_dict(
    torch.load(
        model_path,
        weights_only=True,
        map_location=torch.device("cpu"),
    )
)
model.eval()  # freeze weights
# Init components
picam = Picamera2()
picam.configure(
    picam.create_video_configuration(
        main={
            "format": "RGB888",
            "size": (224, 224),
        },  # WARN: BGR for autopilot
        controls={
            "FrameDurationLimits": (
                int(1_000_000 / params["frame_rate"]),
                int(1_000_000 / params["frame_rate"]),
            )
        },
    )
)
picam.start()
messenger = ThreadedMessenger(port="/dev/ttyACM0", baudrate=115200)
driver = Driver(joy_id=0, autopilot_model=model)

# LOOP
try:
    # frame_counts = 0
    # ave_frame_rate = 0.0
    # start_stamp = time()
    frame_counts = 0
    start_time = time()
    while not driver.is_terminated:
        frame = picam.capture_array()
        frame_counts += 1
        # since_start = time() - start_stamp
        # frame_rate = frame_counts / since_start
        # print(f"frame rate: {frame_rate}")
        mode, st_val, th_val, st_pw, th_pw = driver.process_event(params, frame=frame)
        messenger.out_msg = f"{mode},{st_pw},{th_pw}\n"
        if not frame_counts % params["frame_rate"]:
            elapsed = time() - start_time
            fps = params["frame_rate"] / elapsed
            print("---")
            print(f"Processing at {fps:.2f} FPS | Frame shape: {frame.shape}")
            print(f"steering value: {st_val}, throttle_value: {th_val}")
            print(f"In message (ang_vel_z): {messenger.ang_vel_z}")
            print("Out message: " + messenger.out_msg)
            start_time = time()
        # cv.imshow("Camera", cv.flip(frame, -1))  # picam mounted upside down
        # if cv.waitKey(1) == ord("q"):  # [q]uit
        #     print("Quit signal received.")
        #     break
        # sleep(1 / params["frame_rate"])  # see configs.json for FPS
except KeyboardInterrupt:
    print("\nShutdown signal received.")
finally:
    picam.stop()
