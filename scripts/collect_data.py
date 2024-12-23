import sys
import os
import json
from time import time
from datetime import datetime
import csv
import serial
import pygame
import cv2 as cv
from picamera2 import Picamera2
from gpiozero import LED


# SETUP
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
# Create data directory
image_dir = os.path.join(
    os.path.dirname(sys.path[0]),
    'data', datetime.now().strftime("%Y-%m-%d-%H-%M"),
    'images/'
)
if not os.path.exists(image_dir):
    try:
        os.makedirs(image_dir)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
label_path = os.path.join(os.path.dirname(os.path.dirname(image_dir)), 'labels.csv')
# Init camera
cv.startWindowThread()
cam = Picamera2()
cam.configure(
    cam.create_preview_configuration(
        main={"format": 'RGB888', "size": (224, 224)},
        controls={
            "FrameDurationLimits": (
                int(1000_000 / params['frame_rate']), int(1000_000 / params['frame_rate'])
            )
        },  # 24 FPS
    )
)
cam.start()
for i in reversed(range(3 * params['frame_rate'])):
    frame = cam.capture_array()
    if frame is None:
        print("No frame received. TERMINATE!")
        sys.exit()
    if not i % params['frame_rate']:
        print(i/params['frame_rate'])  # count down 3, 2, 1 sec
# Init variables
ax_val_st = 0. # center steering
ax_val_th = 0. # shut throttle
is_recording = False
frame_counts = 0
frame_rate = 0.
start_stamp = time()

# LOOP
try:
    while True:
        frame = cam.capture_array() # read image
        if frame is None:
            print("No frame received. TERMINATE!")
            break
        for e in pygame.event.get(): # read gamepad events
            if e.type == pygame.JOYAXISMOTION:
                ax_val_st = round((js.get_axis(params['steering_joy_axis'])), 2)  # keep 2 decimals
                ax_val_th = round((js.get_axis(params['throttle_joy_axis'])), 2)  # keep 2 decimals
            elif e.type == pygame.JOYBUTTONDOWN:
                if js.get_button(params['record_btn']):
                    is_recording = not is_recording
                    print(f"Recording: {is_recording}")
                    headlight.toggle()
                elif js.get_button(params['stop_btn']): # emergency stop
                    print("E-STOP PRESSED. TERMINATE!")
                    headlight.off()
                    headlight.close()
                    cv.destroyAllWindows()
                    pygame.quit()
                    ser_pico.close()
                    sys.exit()
        # Calaculate steering and throttle value
        act_st = ax_val_st * params['steering_dir']  # steer action: -1: left, 1: right
        act_th = -ax_val_th  # throttle action: -1: max forward, 1: max backward
        # Encode steering value to dutycycle in nanosecond
        duty_st = params['steering_center'] - params['steering_range'] + \
            int(params['steering_range'] * (act_st + 1))
        # Encode throttle value to dutycycle in nanosecond
        if act_th > 0:
            duty_th = params['throttle_stall'] + \
                int(params['throttle_fwd_range'] * min(act_th, params['throttle_limit']))
        elif act_th < 0:
            duty_th = params['throttle_stall'] + \
                int(params['throttle_rev_range'] * max(act_th, -params['throttle_limit']))
        else:
            duty_th = params['throttle_stall'] 
        msg = (str(duty_st) + "," + str(duty_th) + "\n").encode('utf-8')
        # Transmit control signals
        ser_pico.write(msg)
        # Log data
        action = [act_st, act_th]
        # print(f"action: {action}")  # debug
        if is_recording:
            cv.imwrite(image_dir + str(frame_counts) + '.jpg', frame)
            label = [str(frame_counts) + '.jpg'] + action
            with open(label_path, 'a+', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(label)
        frame_counts += 1
        # Log frame rate
        since_start = time() - start_stamp
        frame_rate = frame_counts / since_start
        print(f"frame rate: {frame_rate}")  # debug
        # Press "q" to quit
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

