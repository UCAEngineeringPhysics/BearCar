from machine import Pin, PWM, reset
from time import sleep

# SETUP
servo = PWM(Pin(17))
servo.freq(50)

# LOOP
for i in range(1000000, 2000000, 10000):
    servo.duty_ns(i)
    print(i)
    sleep(0.2)
for i in reversed(range(1000000, 2000000, 10000)):
    servo.duty_ns(i)
    print(i)
    sleep(0.2)
servo.duty_ns(1500000)
sleep(1)
servo.duty_ns(1800000)
sleep(1)
servo.duty_ns(1500000)
sleep(1)
servo.duty_ns(1200000)
sleep(1)
servo.duty_ns(1500000)
sleep(1)
servo.deinit()
reset()

