from pathlib import Path
import json
from time import sleep, time
from camera import ThreadedCamera
from messenger import ThreadedMessenger
from driver import Driver
import cv2 as cv
import torch

# from torchvision.transforms import v2
from autopilot_architectures.bearnet import BearNet

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
# Load autopilot model
# to_tensor = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
model = BearNet()
model_path = Path(__file__).parents[2].joinpath("models", "dummy.pth")
model.load_state_dict(
    torch.load(
        model_path,
        weights_only=True,
        map_location=torch.device("cpu"),
    )
)
model.eval()  # freeze weights
# Init components
picam = ThreadedCamera()
messenger = ThreadedMessenger(port="/dev/ttyACM0", baudrate=115200)
driver = Driver(joy_id=0, autopilot_model=model)

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
            print(
                f"steering value: {driver.steering_value}, throttle value: {driver.throttle_value}"
            )
            print("Out message: " + messenger.out_msg)
            print(f"In message (ang_vel_z): {messenger.ang_vel_z}")
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
