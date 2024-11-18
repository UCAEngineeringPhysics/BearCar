# Getting Started

## Software Installation
Fire up the terminal on your Raspberry Pi, and run following commands in it.

- Install Dependencies 
```bash
sudo apt install python3-pip
pip install pip --upgrade --break-system-packages
```
- Clone The Repository
```bash
cd ~
git clone https://github.com/UCAEngineeringPhysics/BearCart.git
```
- Install Python Packages
```bash
cd ~/BearCart
pip install -r requirements.txt --break-system-packages
```

## Data Collection (on Raspberry Pi)
```bash
python ~/BearCart/scripts/collect_data.py
```

## Autopilot Model Training (on server)
```bash
python ~/BearCart/scripts/train.py <data directory name in format: year-month-date-hour-minute>
```

## Autopilot Deployment (on Raspberry Pi)
```bash
python ~/BearCart/scripts/autopilot.py
```
