from machine import Pin, PWM
from time import sleep

# SETUP
motor = PWM(Pin(16))
motor.freq(50)
# sleep(3)

# LOOP
for i in range(1500000, 1800000, 10000): # forward up
    motor.duty_ns(i)
    print(i)
    sleep(0.2)
for i in reversed(range(1500000, 1800000, 10000)): # forward down
    motor.duty_ns(i)
    print(i)
    sleep(0.2)
for i in reversed(range(1200000, 1500000, 10000)): # reverse up
    motor.duty_ns(i)
    print(i)
    sleep(0.2)
for i in range(1200000, 1500000, 10000): # reverse down
    motor.duty_ns(i)
    print(i)
    sleep(0.2)
motor.duty_ns(1500000)
sleep(1)
motor.deinit()


