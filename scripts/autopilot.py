import sys
import os
import json
from time import time
import torch
# from torchvision import transforms
from torchvision.transforms import v2
from convnets import BearCartNet
import serial
import pygame
import cv2 as cv
from picamera2 import Picamera2
from gpiozero import LED


# SETUP
# Load configs and init servo controller
model_path = os.path.join(
    os.path.dirname(sys.path[0]),
    'models', 
    'pilot.pth'
)
to_tensor = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
# to_tensor = transforms.ToTensor()
model = BearCartNet()
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()
# Load configs
params_file_path = os.path.join(sys.path[0], 'configs.json')
with open(params_file_path, 'r') as file:
    params = json.load(file)
# Init LED
headlight = LED(params['led_pin'])
headlight.off()
# Init serial port
ser_pico = serial.Serial(port='/dev/ttyACM0', baudrate=115200)
print(f"Pico is connected to port: {ser_pico.name}")
# Init controller
pygame.display.init()
pygame.joystick.init()
js = pygame.joystick.Joystick(0)
# init camera
cv.startWindowThread()
cam = Picamera2()
cam.configure(
    cam.create_preview_configuration(
        main={"format": 'RGB888', "size": (224, 224)},
        controls={"FrameDurationLimits": (41667, 41667)},  # 24 FPS
    )
)
cam.start()
for i in reversed(range(72)):
    frame = cam.capture_array()
    # cv.imshow("Camera", frame)
    if frame is None:
        print("No frame received. TERMINATE!")
        sys.exit()
    if not i % 24:
        print(i/24)  # count down 3, 2, 1 sec
# Init timer for FPS computing
start_stamp = time()
frame_counts = 0
ave_frame_rate = 0.
# Init variables
is_paused = True


# LOOP
try:
    while True:
        frame = cam.capture_array()  # read image
        if frame is None:
            print("No frame received. TERMINATE!")
            break
        for e in pygame.event.get():  # read controller input
            if e.type == pygame.JOYBUTTONDOWN:
                if js.get_button(params['record_btn']):
                    is_paused = not is_paused
                    print(f"Paused: {is_paused}")
                    headlight.toggle()
                elif js.get_button(params['stop_btn']): # emergency stop
                    print("E-STOP PRESSED. TERMINATE!")
                    headlight.off()
                    headlight.close()
                    cv.destroyAllWindows()
                    pygame.quit()
                    ser_pico.close()
                    sys.exit()
        # predict steer and throttle
        img_tensor = to_tensor(frame)
        with torch.no_grad():
            pred_st, pred_th = model(img_tensor[None, :]).squeeze()
        st_trim = float(pred_st)
        if st_trim >= 1:  # trim steering signal
            st_trim = .999
        elif st_trim <= -1:
            st_trim = -.999
        th_trim = float(pred_th)
        if th_trim >= 1:  # trim throttle signal
            th_trim = .999
        elif th_trim <= -1:
            th_trim = -.999
        # Encode steering value to dutycycle in nanosecond
        if is_paused:
            duty_st = params['steering_center']
        else:
            duty_st = params['steering_center'] - params['steering_range'] + \
                int(params['steering_range'] * (st_trim + 1))
        # Encode throttle value to dutycycle in nanosecond
        if is_paused:
            duty_th = params['throttle_stall']
        else:
            if th_trim > 0:
                duty_th = params['throttle_stall'] + \
                    int(params['throttle_fwd_range'] * min(th_trim, params['throttle_limit']))
            elif th_trim < 0:
                duty_th = params['throttle_stall'] + \
                    int(params['throttle_rev_range'] * max(th_trim, -params['throttle_limit']))
            else:
                duty_th = params['throttle_stall']
        msg = (str(duty_st) + "," + str(duty_th) + "\n").encode('utf-8')
        # Transmit control signals
        ser_pico.write(msg)
        print(f"predicted action: {pred_st, pred_th}")  # debug
        frame_counts += 1
        # Log frame rate
        since_start = time() - start_stamp
        frame_rate = frame_counts / since_start
        print(f"frame rate: {frame_rate}")  # debug
        if cv.waitKey(1)==ord('q'):
            print("Quit signal received.")
            break
 
# Take care terminate signal (Ctrl-c)
except KeyboardInterrupt:
    headlight.off()
    headlight.close()
    cv.destroyAllWindows()
    pygame.quit()
    ser_pico.close()
    sys.exit()
finally:
    headlight.off()
    headlight.close()
    cv.destroyAllWindows()
    pygame.quit()
    ser_pico.close()
    sys.exit()
