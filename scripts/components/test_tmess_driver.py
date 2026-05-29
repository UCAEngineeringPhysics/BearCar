import sys
from pathlib import Path
import json
from time import sleep
from messenger import ThreadedMessenger
from driver import Driver

# SAFETY CHECK
is_tangled = input("Observe BearCar closely! Is any wire touching the wheels? (Y/n)")
while is_tangled != "n":
    print("Please decouple the wires from BearCar's wheels!")
    is_tangled = input(
        "Observe BearCar closely! Is any wire touching the wheels? (Y/n)"
    )
print("~~~\nGet ready, Human!\n~~~")
for i in reversed(range(3)):
    print(i + 1)
    sleep(1)

# SETUP
# Load configs
params_file_path = Path(__file__).parent.joinpath("configs.json")
with open(params_file_path, "r") as file:
    params = json.load(file)
# Init serial port
messenger = ThreadedMessenger(port="/dev/ttyACM0", baudrate=115200)
driver = Driver(joy_id=0, autopilot_model=None)

# LOOP
try:
    while not driver.is_terminated:
        mode, st_val, th_val, st_pw, th_pw = driver.process_event(params, frame=None)
        messenger.out_msg = f"{mode},{st_pw},{th_pw}\n"
        print("---")
        print(f"steering value: {st_val}, throttle_value: {th_val}")
        print(f"In message (ang_vel_z): {messenger.ang_vel_z}")
        print("Out message: " + messenger.out_msg)
        # 50Hz
        sleep(0.02)
except KeyboardInterrupt:
    sys.exit()
