# Autopilot Training Guide
The autopilot is a deep learning model.
To be more specific, it is a convolutional neural network model.
This model takes in images captured by the camera, then predict driving actions (throttle and steering) accordingly.

To train such a model, we need to feed human operators' driving data to it.
The trained model is expected to imitate its human teachers' driving behaviors.
Basically, this is a behavioral cloning approach.

Any computer with PyTorch installed can be used to train such a model.
So, theoretically, the Raspberry Pi can be used for the training purpose.
However, we strongly recommend you to setup a dedicated model training server to train your autopilot model.
For the computational power of a Raspberry Pi is quite limited.

The following steps will guide you through the entire process of autopilot training on a server computer.
You can perform all the operations on the Raspberry Pi without physically interact with the server computer.

*For those intend to train models on your Raspberry Pis, please ignore the data transferring steps.*

The following steps assume that your server is sharing the same network as the Raspberry Pi.
So, the first 3 segments in the ip address on both machines should be the same.
In our case, they both start with `192.168.0.`.

## 1 Transfer Collected Data to Server

- Usage:
```bash
rsync -rv --progress --partial <data_path_on_rpi> <username>@<server_ip_address>:<data_path_on_server>
```

- Example:
```bash
rsync -rv --progress --partial ~/BearCart/data/2024-08-13-15-05 user@192.168.0.123:~/BearCart/data/  # example
```

## 2 Run Training Script
### 2.1 Remotely Log In to Server via SSH

- Usage:
```bash
ssh <username>@<server_ip_address>
```

- Example:
```bash
ssh user@192.168.0.123
```
You'll need to enter the correct password to log in.

### 2.2 Training with Specified Data

- Usage:
```bash
python ~/BearCart/scripts/train.py <data_directory_name> 
```

- Example:
```bash
python ~/BearCart/scripts/train.py 2024-08-13-15-05 
```
The trained model will be saved at:
```bash
<model_path>/DonkeyNet-15epochs-0.001lr.pth
```
For example:
```bash
~/BearCart/data/2024-08-13-15-05/DonkeyNet-15epochs-0.001lr.pth
```
And, the model is stored on the server computer now.

Now you can log out from the server by typing `exit` in terminal.
Or press `Ctrl`+`d` on keyboard.

## 3 Transfer Autopilot Model to Raspberry Pi
**Make sure you are back to Raspberry Pi.**

- Usage:
```bash
rsync -rv --progress --partial <username>@<server_ip_address>:<model_path_on_server> <model_dir_path>
```

-Example:
```bash
rsync -rv --progress --partial user@192.168.0.123:~/BearCart/data/2024-08-13-15-05/DonkeyNet-15epochs-0.001lr.pth ~/BearCart/models/
```



