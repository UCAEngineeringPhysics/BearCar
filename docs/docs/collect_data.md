# Data Collection Guide

## Script Execution
In a new RPi's terminal, run following command.
```console
python ~/BearCart/scripts/collect_data.py
```

## Usage

### Toggle Recording
The default button for toggling data recording is `R1`.
Headlight will be toggled as well if it is correctly wired up.
Press the button to activate/deactivate data recording. 

### Emergency Stop 
The default button for EMERGENCY STOP is `X`.
Press this button will terminate the execution of the Python script.
Release throttle (right) joystick if a normal stop is expected.


## Saved Data
A data directory will be automatically created once the `collect_data.py` script is executed.
The data directory will be located at `~/BearCart/data/<year-month-day-hour-minute>`.
And it will be timestamped with the moment of executing the `collect_data.py` script.

> If toggle recording button was never pressed, the data directory will be empty.

If data was effectively recorded, extra files will be saved in the timestamped data directory.
An example of the data structure looks like the follows.
```console
BearCart/data/2024-08-13-15-05/
├── images/
└── labels.csv
```
The `images` directory stores all the images recorded during the data collection process.
The saved images will be indexed follow the chronological order. 
The `labels.csv` file stores the "image file names", "human input steering values" and "human input throttle values".
These files are organized in 3 columns. 
And each row stands for the human operator's action at the moment of the corresponding image was captured.




