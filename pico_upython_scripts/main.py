"""
Upload this script to the pico board, then rename it to main.py.
"""

import sys
import select
from time import sleep
from machine import Pin, PWM, WDT, freq, reset

# SETUP
# Overclock
freq(200_000_000)  # Pico 2 original: 150_000_000
# Config LED pins
r_led = Pin(18, Pin.OUT)
g_led = Pin(19, Pin.OUT)
b_led = Pin(20, Pin.OUT)
# Config drivetrain pins
steering = PWM(Pin(17))
steering.freq(50)
steering.duty_ns(1_500_000)
throttle = PWM(Pin(16))
throttle.freq(50)
throttle.duty_ns(1_500_000)
# TODO: Config LED pins
# Config USB BUS
listener = select.poll()
listener.register(sys.stdin, select.POLLIN)
event = listener.poll()
# print("Pico listening...")  # uncomment to debug
# Config watchdog timer
wdt = WDT(timeout=500)  # ms
# Variables
mode = 'p'

# LOOP
try:
    while True:
        dutycycle_st = 1_500_000
        dutycycle_th = 1_500_000
        for msg, _ in event:
            if msg:
                wdt.feed()
                msg_line = msg.readline().rstrip()
                print(f"Pico heard: {msg_line}")  # debug
                msg_parts = msg_line.split(",")
                # print(f"Pico heard: {msg_parts}")  # debug
                if len(msg_parts) == 3:
                    # print("Pico heard 2 parts")  # debug
                    try:
                        # Process driving actions
                        dutycycle_st = int(msg_parts[1])
                        dutycycle_th = int(msg_parts[2])
                        # Process LED actions
                        mode = msg_parts[0]
                        if mode is 'n':  # normal
                            r_led.value(0)
                            g_led.value(1)
                            b_led.value(0)
                        elif mode is 'r':  # recording
                            r_led.value(0)
                            g_led.value(0)
                            b_led.value(1)
                        elif mode is 'a':  # autopilot
                            r_led.value(1)
                            g_led.value(0)
                            b_led.value(1)
                        elif mode is 'p':  # pause
                            r_led.value(1)
                            g_led.value(0)
                            b_led.value(0)
                            dutycycle_st = 1_500_000
                            dutycycle_th = 1_500_000
                        else:
                            r_led.value(0)
                            g_led.value(0)
                            b_led.value(0)
                            dutycycle_st = 1_500_000
                            dutycycle_th = 1_500_000
                        print(f"Pico received command: {mode}, {dutycycle_st}, {dutycycle_th}") # debug
                        steering.duty_ns(dutycycle_st)
                        throttle.duty_ns(dutycycle_th)
                    except ValueError:
                        # print("ValueError!")  # debug
                        reset()
            else:
                throttle.duty_ns(0)
                steering.duty_ns(0)
                reset()
except Exception as e:
    # print('Pico reset')  # debug
    reset()
finally:
    # print('Pico reset')  # debug
    reset()
