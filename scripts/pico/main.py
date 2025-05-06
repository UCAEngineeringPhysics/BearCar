"""
Upload this script to the pico board, then rename it to main.py.
"""
import sys
import select
from time import sleep, ticks_ms, ticks_diff
from machine import Pin, PWM, WDT, freq, reset

# SETUP
# Overclock
freq(200_000_000)  # Pico 2 original: 150_000_000
# Config Pins
steering = PWM(Pin(16))
steering.freq(50)
steering.duty_ns(0)
thruster = PWM(Pin(21))
thruster.freq(50)
thruster.duty_ns(0)
sleep(3)  # ESC calibration. TODO: investigate how
# Config USB BUS
listener = select.poll()
listener.register(sys.stdin, select.POLLIN)
event = listener.poll()
# print("I'm listening...")  # uncomment to debug
# Config watchdog timer
wdt = WDT(timeout=500)  # ms

# LOOP
try:
    while True:
        dutycycle_st = 0
        dutycycle_th = 0
        for msg, _ in event:
            if msg:
                wdt.feed()
                msg_line = msg.readline().rstrip()
                # print(f"Pico heard: {msg_line}")  # debug
                msg_parts = msg_line.split(',')
                # print(f"Pico heard: {msg_parts}")  # debug
                if len(msg_parts) == 2:
                    # print("Pico heard 2 parts")  # debug
                    try:
                        dutycycle_st = int(msg_parts[0])
                        dutycycle_th = int(msg_parts[1])
                        # print(f"Pico received dutycycle: {dutycycle_st}, {dutycycle_th}") # debug
                        steering.duty_ns(dutycycle_st)
                        thruster.duty_ns(dutycycle_th)
                    except ValueError:
                        # print("ValueError!")  # debug
                        reset()
            else:
                thruster.duty_ns(0)
                steering.duty_ns(0)
                reset()
except Exception as e:
    # print('Pico reset')  # debug
    thruster.duty_ns(0)
    steering.duty_ns(0)
    reset()
finally:
    # print('Pico reset')  # debug
    thruster.duty_ns(0)
    steering.duty_ns(0)
    reset()
