# Software Installation Guide

## 1. Software on Pico
[Pico](https://www.raspberrypi.com/documentation/microcontrollers/pico-series.html) is a microcontroller.
It bridges the Raspberry Pi and the drivetrain (ESC and Servo) of BearCart.
Raspberry Pi sends desired propelling and turning values to Pico.
Pico translate them into the PWM signals then send them to the drivetrain.

### 1.1 Upload MicroPython Firmware to Pico
The MicroPython firmware allows Pico to execute Python code.
Please follow the instructions in the [Getting started with Raspberry Pi Pico](https://projects.raspberrypi.org/en/projects/getting-started-with-the-pico/3) guide.

### 1.2 Calibrate ESC and Servo
Make sure you have followed the [Wiring Guide](wiring.md) to connect Pico to Raspberry Pi, ESC and servo.
If using Thonny, remember to set your Python interpreter to MicroPython at the bottom right corner.

1. Run [esc_servo_test.py](https://github.com/UCAEngineeringPhysics/BearCart/blob/671e7794f572fbdda4885c78627b4d613b9486a2/scripts/pico/esc_servo_test.py).
Take down the duty cycle values steer the front wheels all the way to the left, all way to the right and middle.
2. Run [esc_throttle_test.py](https://github.com/UCAEngineeringPhysics/BearCart/blob/671e7794f572fbdda4885c78627b4d613b9486a2/scripts/pico/esc_throttle_test.py).
**CAUTION: lift the BearCart up, and make sure nothing has contact with any tire!**

Take down the duty cycle values/ranges that spin the throttle motor forward at its maximum speed, spin it reversely at its maximum speed, and stop it.

### 1.3 Upload the Command Listener Script to Pico
Upload or save [main.py](https://github.com/UCAEngineeringPhysics/BearCart/blob/671e7794f572fbdda4885c78627b4d613b9486a2/scripts/pico/main.py) to Pico.
If using Thonny, you'll notice the file name becomes `[main.py]` if the script is successfullly saved on board.


