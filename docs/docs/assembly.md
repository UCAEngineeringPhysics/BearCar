# Hardware Assembly Guide
We recommend you to follow the following order to assemble your BearCart.

## 1 Standoffs
| Item                          | Qty.  |
| :---                          | :---  |
| Male M2.5*15 standoff         | 8     |
| M2.5 nut                      | 8     |
| Female M2*15 standoff         | 4     |
| M2*6 screw                    | 4     |

There are pockets at the bottom side of the bed. 
Place nuts or screws into those pockets and tighten the standoffs by hands.

![prep_standoff](images/hardware/prep_standoff.jpg)
![prep_camount](images/hardware/prep_camount.jpg)

## 2 Camera
| Item                    | Qty.  |
| :---                    | :---  |
| 3D-Printed camera mount | 1     |
| RPi Camera module       | 1     |
| CSI **camera** cable        | 1     |
| M2.5*6 screw            | 2-4   |
| M2.5 nut                | 2-4   |
| M2*6 screw              | 2-4   |
| M2 nut                | 2-4   |

### 2.1 Install Camera Mount on Bed
![prep_camount](images/hardware/prep_camount.jpg)
![post_camount](images/hardware/post_camount.jpg)
![post_standoff_back](images/hardware/post_standoff_back.jpg)

### 2.2 Install Pi Camera
![prep_cam](images/hardware/prep_cam.jpg)
![post_cam_front](images/hardware/post_cam_front.jpg)
![post_cam_back](images/hardware/post_cam_back.jpg)


## 3 Wire Splitter
| Item                             | Qty.  |
| :---                             | :---  |
| Wire splitter                    | 1     |
| M2.5*16 screw                    | 2     |
| M2.5 nut                         | 2     |
| Female T-plug connector w/ wires | 1     |
| Male T-plug connector w/ wires   | 1     |
| Female JST connector w/ wires    | 1     |

![prep_splitter](images/hardware/prep_splitter.jpg)
![post_splitter](images/hardware/post_splitter.jpg)
![post_splitter_bed](images/hardware/post_splitter_bed.jpg)

## 4 Buck Converter
| Item                          | Qty.  |
| :---                          | :---  |
| Step-Down (BUCK) converter    | 1     |
| M2.5*6 screw                  | 4     |
| Male JST connector w/ wires   | 1     |
| Female JST connector w/ wires | 1     |

**Note the arrow pointing direction below the LCD screen**

![prep_converter](images/hardware/prep_converter.jpg)
![post_converter](images/hardware/post_converter.jpg)
![post_converter_bed](images/hardware/post_converter_bed.jpg)

## 5 Pico
| Item                              | Qty.  |
| :---                              | :---  |
| RPi Pico                          | 1     |
| M2*6 screw                        | 2-4   |
| Male-to-Male Dupont jumper wire   | 1     |
| Male-to-Female Dupont jumper wire | 4     |
| Micro-USB to USB-A cable          | 1     |
| Double-Sided tape                 | 1     |

Pico's `GP0`, `GP15` and 2 `GND` pins will be employed later to control the ESC and the servo motor. Please refer to the [Wiring Guide](wiring.md) for more details.
![prep_pico](images/hardware/prep_pico.jpg)
![post_pico](images/hardware/post_pico.jpg)

## 6 Raspberry Pi
| Item                              | Qty.  |
| :---                              | :---  |
| Raspberry Pi 5                    | 1     |
| M2.5*6 screw                      | 2-4   |

![prep_pi](images/hardware/prep_pi.jpg)
![post_pi](images/hardware/post_pi.jpg)

## 7 Replace ESC
| Item                     | Qty.  |
| :---                     | :---  |
| QuicRun 1060 brushed ESC | 1     |
| M2*6 screw               | 2     |
| M2 nut                   | 2     |
| Double-Sided tape        | 2     |

1. Unplug motor power wires (blue and yellow) from the stock ESC.
![stock_esc](images/hardware/stock_esc.jpg)
2. Plug the motor power wires to blue and yellow wires on the new ESC
![esc_motor](images/hardware/esc_motor.jpg)
3. The new ESC can be seated on top of the servo motor.
![esc_spot](images/hardware/esc_spot.jpg)
4. Wire up ESC, servo motor and Pico
![esc-servo-pico](images/hardware/esc-servo-pico.jpg)
For a more detailed guide on wiring, please go to the [Wiring Guide](wiring.md) page.



