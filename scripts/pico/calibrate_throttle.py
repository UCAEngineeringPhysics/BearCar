from machine import Pin, PWM, reset
from time import sleep

print("Please calibrate your ESC follow the steps below:")
print("1. Turn off ESC")
print("2. Unplug Pico")
print("3. Plug Pico back in, and turn ESC on.")
print("4. Run this MicroPython script.")

# SETUP
throttle = PWM(Pin(16))
throttle.freq(50)
# throttle.duty_ns(0)

# LOOP
try:
    throttle.duty_ns(1500000)
    sleep(1)
    throttle.duty_ns(1800000)  # to make sure
    sleep(1)
    throttle.duty_ns(1500000)
    sleep(2)
except:
    print("Exception!")
finally:
    reset()
print("Throttle is calibrated if you heard a long beep, followed by 2 short beeps.")
print("If not, repeat and follow the instructions carefully.")
