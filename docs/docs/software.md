# Software Installation  and Test Guide

## 1. Software on Pico
[Pico](https://www.raspberrypi.com/documentation/microcontrollers/pico-series.html) is a microcontroller.
It bridges the Raspberry Pi and the drivetrain (ESC and servo) of BearCart.
Raspberry Pi sends desired propelling and turning commands to Pico.
Pico translates them into the PWM signals then sends them to the drivetrain.
The code running on Pico is located at [BearCart/scripts/pico/](https://github.com/UCAEngineeringPhysics/BearCart/tree/main/scripts/pico).

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
If using Thonny, you'll notice the file name becomes `[main.py]`.

## 2. Software on Raspberry Pi

### 2.1 BearCart Software Installation
Open Raspberry Pi's terminal window and follow the steps below:
#### 2.1.1 Install dependencies:
```bash
sudo apt install python3-pip
pip install pip --upgrade --break-system-packages
```
#### 2.1.2 Download BearCart repository
```bash
cd ~
git clone https://github.com/UCAEngineeringPhysics/BearCart.git
```
#### 2.1.3 Install Python packages
```bash
cd ~/BearCart
pip install -r requirements.txt --break-system-packages
```

### 2.2 Update Configurations
Open up [configs.json](https://github.com/UCAEngineeringPhysics/BearCart/blob/671e7794f572fbdda4885c78627b4d613b9486a2/scripts/configs.json) in a text editor.

- Change the follwing lines according to servo calibration results
```json
"steering_left": 1000000,
"steering_right": 2000000,
"steering_center": 1500000,
"steering_range": 500000,
```
- Change the following lines according to throttle calibration results
```json
"throttle_stall": 1210000,
"throttle_fwd_range": 590000,
"throttle_rev_range": 120000,
```

### 2.3 Unit Test
All the unit testing scripts are located at [BearCart/scripts/unit_test](https://github.com/UCAEngineeringPhysics/BearCart/tree/main/scripts/unit_test)

1. Test Raspberry Pi camera module: [camera.py](https://github.com/UCAEngineeringPhysics/BearCart/blob/671e7794f572fbdda4885c78627b4d613b9486a2/scripts/unit_test/camera.py).
> **CAUTION: power down Raspberry Pi before connect/disconnect the camera**

2. Test gamepad: [joystick.py](https://github.com/UCAEngineeringPhysics/BearCart/blob/671e7794f572fbdda4885c78627b4d613b9486a2/scripts/unit_test/joystick.py).
> Make sure the gamepad is connected via bluetooth.

3. Test steering command transmitting: [serial_steering.py](https://github.com/UCAEngineeringPhysics/BearCart/blob/671e7794f572fbdda4885c78627b4d613b9486a2/scripts/unit_test/serial_steering.py).

4. Test throttle command transmitting: [serial_throttle.py](https://github.com/UCAEngineeringPhysics/BearCart/blob/671e7794f572fbdda4885c78627b4d613b9486a2/scripts/unit_test/serial_throttle.py).

5. Integration test: [camera_joystick_drivetrain.py](https://github.com/UCAEngineeringPhysics/BearCart/blob/671e7794f572fbdda4885c78627b4d613b9486a2/scripts/unit_test/camera_joystick_drivetrain.py).
