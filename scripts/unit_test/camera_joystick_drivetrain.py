"""
Integrated test with controller, pico usb communication, throttle motor.
"""
import sys
import os
import json
from time import time
import serial
import pygame
import cv2 as cv
from picamera2 import Picamera2
from gpiozero import LED

# SETUP
# Load configs
params_file_path = os.path.join(os.path.dirname(sys.path[0]), 'configs.json')
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
# Init Pi Camera
picam = Picamera2()
picam.configure(
    picam.create_preview_configuration(
        main={"format": 'RGB888', "size": (224, 224)},
        controls={
            "FrameDurationLimits": (
                int(1000_000 / params['frame_rate']), int(1000_000 / params['frame_rate'])
            )
        },  # 24 FPS
    )
)
picam.start()
for i in reversed(range(3 * params['frame_rate'])):
    frame = picam.capture_array()
    if frame is None:
        print("No frame received. TERMINATE!")
        sys.exit()
    if not i % params['frame_rate']:
        print(i/params['frame_rate'])  # count down 3, 2, 1 sec
# Init timer for FPS computing
start_stamp = time()
frame_counts = 0
ave_frame_rate = 0.
# Init joystick axes values
ax_val_st = 0.
ax_val_th = 0.

# MAIN LOOP
try:
    while True:
        frame = cam.capture_array() # read image
        if frame is None:
            print("No frame received. TERMINATE!")
            break
        cv.imshow('camera', frame)
        for e in pygame.event.get(): # read controller input
            if e.type == pygame.JOYAXISMOTION:
                ax_val_st = round((js.get_axis(params['steering_joy_axis'])), 2)  # keep 2 decimals
                ax_val_th = round((js.get_axis(params['throttle_joy_axis'])), 2)  # keep 2 decimals
            elif e.type == pygame.JOYBUTTONDOWN:
                if js.get_button(params['record_btn']):
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
        act_th = -ax_val_th # throttle action: -1: max forward, 1: max backward
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
        ser_pico.write(msg)
        # Log action
        action = [act_st, act_th]
        print(f"action: {action}")
        # Log frame rate
        frame_counts += 1
        since_start = time() - start_stamp
        frame_rate = frame_counts / since_start
        print(f"frame rate: {frame_rate}")
        # Wait for [Q]uit signal
        if cv.waitKey(1)==ord('q'):
            print("Quit signal received.")
            break

# Take care terminal signal (Ctrl-c)
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
