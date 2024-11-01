# Electric Wiring Guide
**Color coding the positive and negative power lines is extremely recommended.**

## 1 Split Battery Power
The LiPo battery will provide power for all the electric components. 
A wire splitter is employed to separate the power for the drivetrain from the power for the computing devices.


![battery-splitter](images/wiring/battery-splitter.png)

## 2 Powering Drivetrain
The Electronic Speed Controller (ESC) is in charge of regulate the speed and direction of the main engine (a DC motor), as well as providing power to the steering system (a servo motor) through a Battery Eliminator Circuit (BEC).

![esc-motor-servo](images/wiring/esc-motor-servo.png)

The ESC and the servo motor can receive signals through their JST connectors.
There are two widely used color codes for these connections.

![servo_color](https://i0.wp.com/dronebotworkshop.com/wp-content/uploads/2018/05/servo-motor-pinout.jpg?w=768&ssl=1)

## 3 Powering Computers
The computing devices are strict on the power sources. We use a step-down converter (BUCK converter) to lower the battery output voltage (7.4V nominal) to 5V. 

![converter-pi-pico](images/wiring/buck-pi-pico.png)

## 5 GPIO Wiring
- Raspberry Pi is in charge of controlling the steering servo and thrust motor.
- The control signals are handled by some of the GPIO pins.
- The GPIO pins employed as shown in the diagram below are the default setting.
Other GPIO pins can be used.

![gpio](/_DOCS/assemble/electric/images/gpio.jpg)

## TODO
- LED wiring

