# Bill of Materials

## Part List
The total cost for building a BearCart from the listed items below is around $250.
Although, you can always find cheaper replacement for pretty much every single item in the list.

| Item         | Description                     | Qty.  | Unit Price   |
| :---:        | :---                            | :---: |         ---: |
| [MEW4 1/16 RC Car](https://www.amazon.com/dp/B0CKPCPDSG) |  Powered by brushed 390 motor | 1 | $98.88 |
| [Raspberry Pi 5](https://www.pishop.us/product/raspberry-pi-5-4gb/?src=raspberrypi)<sup>1</sup> | SBC for running AI autopilot and other Python scripts | 1 | $60.00 |
| [RPi Camera V2](https://www.amazon.com/dp/B01ER2SKFS?th=1)<sup>2</sup> | Eye of the BearCart, 8-megapixel | 1 | $12.50 |
| [Camera Cable](https://www.pishop.us/product/camera-cable-for-raspberry-pi-5/)<sup>3</sup> | 200mm 22 pin to 15 pin | 1 | $1.20 |
| [Micro-HDMI to Standard HDMI Cable](https://www.pishop.us/product/micro-hdmi-to-standard-hdmi-a-m-1m-cable-white/) | Connect monitor to RPi 5 | 1 | $5.56 |
| [MicroSD Card](https://www.amazon.com/dp/B09X7C7LL1/?th=1) | 64GB, U3, A2 | 1 | $11.27 |
| [Raspberry Pi Pico](https://www.pishop.us/product/raspberry-pi-pico-h-pre-soldered-headers/) | Microcontroller listens to RPi commands and controls drivetrain | 1 | $5.00 |
| [Micro USB Cable](https://www.amazon.com/dp/B07G934SJ9) | Micro-USB to USB-A, data capable | 1 | $2.67 |
| [QUICRUN 1060 ESC](https://www.amazon.com/dp/B0CM876HQB/)<sup>4</sup> | 60 Amp continuous current and a 6V3A BEC integrated | 1 | $21.99 |
| [Wireless Game Pad](https://www.amazon.com/gp/product/B0BWH7FBZC/)<sup>5</sup> | Bluetooth | 1 | $19.99 |
| [Adjustable Buck Converter](https://www.amazon.com/dp/B0CXT83GV6)<sup>6</sup> | 6V - 32V to 1.5V - 32V | 1 | $12.49 |
| [Lever Wire Splitter](https://www.amazon.com/dp/B08JPBJDW4) | 2-in, 4-out | 1 | $1.59 |
| [T-Plug Connector](https://www.amazon.com/dp/B07WHPD4KD/) | One pair of male and female, bridges battery and ESC | 1 | $1.60 |
| [Male JST Plug Connector Wires](https://www.amazon.com/dp/B01M5AHF0Z) | Connect voltage converter and RPi | 1 | $0.33 |
| [Female JST Plug Connector Wires](https://www.amazon.com/dp/B01M5AHF0Z) | Connect voltage converter and RPi | 2 | $0.33 |
| [Dupont Jumper Wires, Male to Female](https://www.amazon.com/dp/B07GCZVCGS) | Connect Pico and drivetrain | 4 | < $0.1  |
| [Dupont Jumper Wires, Male to Male](https://www.amazon.com/dp/B07GCZVCGS) | Connect Pico and drivetrain | 4 | < $0.1 |
| [M2*15 Female Standffs](https://www.amazon.com/dp/B0BP6MT7RP) | Attach Pico to the bed | 4 | < $0.1 |
| [M2*6 Screws ](https://www.amazon.com/dp/B0BP6MT7RP) | Secure units | 14 | < $0.1  |
| [M2 Nuts ](https://www.amazon.com/dp/B0BP6MT7RP) | Secure units | 6 | < $0.1 |
| [M2.5*15 Male-Female Standoffs](https://www.amazon.com/dp/B0BP6LT76V) | Lift PCBs up | 8 | < $0.1 |
| [M2.5 Nuts](https://www.amazon.com/dp/B0CLM8BFQX/?th=1) | secure screws and standoffs on the bed | 16 | < $0.1 |
| [M2.5*6 Screws](https://www.amazon.com/dp/B0CLM8BFQX/?th=1) | Attach PCBs to the bed | 12 | < $0.1 |
| [M2.5*16 Screws](https://www.amazon.com/dp/B0CLM8BFQX/?th=1) | Attach wire splitter to the bed | 2 | < $0.1 |
| [3D Printed Bed](https://github.com/UCAEngineeringPhysics/BearCart/blob/mech/mechanical_designs/ucaep_mew4_bed_v3.stl) | Hosts above components | 1 | ~ $1.21 |
| [3D Printed Camera Mount](https://github.com/UCAEngineeringPhysics/BearCart/blob/mech/mechanical_designs/CameraMount.stl) | Holds RPi camera | 1 | < $0.1 |

<sup>1</sup>: 8GB RAM and accessories are optional

<sup>2</sup>: V1, V3 or V4 cameras are fine

<sup>3</sup>: Don't be tricked by the [**Display** Cable](https://www.pishop.us/product/display-cable-for-raspberry-pi-5/)

<sup>4</sup>: Electronic speed controller. Other brands and products are OK, but make sure the unit **DOES NOT** have radio receiver (antenna) integrated.

<sup>5</sup>: Any generic gamepad on the market should work. Get a bluetooth one if you want your BearCart and the gamepad to be one-on-one tethered (You'll want this when you are managing multiple BearCarts).

<sup>6</sup>: Get a dedicated [DC-DC Buck converter](https://www.dfrobot.com/product-2162.html) with 5V/8A output, if not interested in monitoring the battery health or need more space on the board.

## Peripherals

| Item     | Requirement   | Qty.  |
| :---:    | :---          | :---: |
| Monitor  | Has HDMI port | 1     |
| Keyboard | USB connector | 1     |
| Mouse    | USB connector | 1     |

