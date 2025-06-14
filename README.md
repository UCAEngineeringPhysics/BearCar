# BearCar

BearCar is an autonomous driving platform based on an 1:16 RC car and Raspberry Pi.
Visit [documentations](https://ucaengineeringphysics.github.io/BearCar/) for more details.

This project is strongly inspired by the 
[DonkeyCar](https://github.com/autorope/donkeycar) project.


## Quick Start
Fire up the terminal on your Raspberry Pi, and run following commands in it.

### Download BearCar Project
```bash
cd ~
git clone https://github.com/UCAEngineeringPhysics/BearCar.git
```

### Setup Environment
```bash
cd ~/BearCart
./setup_pi_env.sh
```

### Have Fun Racing!
```bash
cd ~/BearCart
uv run scripts/teleop.py
```

## Demo Videos
- [Initial BearCart](https://youtube.com/shorts/Kcm6qQqev3s)
- [Another Autopilot](https://youtu.be/8GX6HnfgrJQ)
