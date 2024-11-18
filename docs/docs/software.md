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
Make sure you have followed the wiring guide  



