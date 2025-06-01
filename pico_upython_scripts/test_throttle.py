from machine import Pin, PWM, reset
from time import sleep

# SAFETY CHECK
is_lifted = input("Is something contacting any wheel of BearCart? (Y/n)")
while is_lifted is not "n":
    print("Please lift BearCart up and remove everything that is making the contact")
    is_lifted = input("Is something contacting any wheel of BearCart? (Y/n)")
print("Hold tight! Unleash the beast!!!")

# SETUP
throttle = PWM(Pin(16))
throttle.freq(50)
# TODO: config led for a naive HRI

# LOOP
print("\nFORWARD: ramp up\n")
for dc in range(1500000, 2000000, 10000): # forward up
    throttle.duty_ns(dc)
    print(dc)
    sleep(0.2)
print("\nFORWARD: ramp down\n")
for dc in reversed(range(1500000, 2000000, 10000)): # forward down
    throttle.duty_ns(dc)
    print(dc)
    sleep(0.2)
print("\nREVERSE: ramp up\n")
for dc in reversed(range(1000000, 1500000, 10000)): # reverse up
    throttle.duty_ns(dc)
    print(dc)
    sleep(0.2)
print("\nREVERSE: ramp down\n")
for dc in range(1000000, 1500000, 10000): # reverse down
    throttle.duty_ns(dc)
    print(dc)
    sleep(0.2)
throttle.duty_ns(1500000)
print("STOP")
sleep(1)
throttle.deinit()
reset()

