"""
This script will test communication between Pico and RPi.
Please uncomment line 45 in the 'main.py' on Pico.
"""

from serial import Serial
from time import sleep

# SAFETY CHECK
is_lifted = input("Is something contacting any wheel of BearCart? (Y/n)")
while is_lifted != "n":
    print("Please lift BearCart up and remove everything that is making the contact")
    is_lifted = input("Is something contacting any wheel of BearCart? (Y/n)")
print("Hold tight! You are about to unleash the beast!")
print("Verify light order: cyan -> yellow -> green -> blue -> purple -> red")
for i in reversed(range(3)):
    print(i + 1)
    sleep(1)

# SETUP
messenger = Serial(port="/dev/ttyACM0", baudrate=115200, timeout=0.01)
print(messenger.name)
# Constants
modes = ("s", "p", "n", "r", "a", "e")
pulsewidths = (1_500_000, 1_600_000, 1_400_000)

# LOOP
sleep(3)  # Stablize communication
messenger.write(
    f"{modes[0]},{pulsewidths[0]},{pulsewidths[0]}\n".encode("utf-8")
)  # standby
if messenger.inWaiting() > 0:
    reply = messenger.readline()
    reply = reply.decode("utf-8", "ignore")
    print(f"[Pico message]: {reply}")
sleep(2)
messenger.write(
    f"{modes[1]},{pulsewidths[0]},{pulsewidths[0]}\n".encode("utf-8")
)  # pause
if messenger.inWaiting() > 0:
    reply = messenger.readline()
    reply = reply.decode("utf-8", "ignore")
    print(f"[Pico message]: {reply}")
sleep(2)
messenger.write(
    f"{modes[2]},{pulsewidths[1]},{pulsewidths[1]}\n".encode("utf-8")
)  # normal
if messenger.inWaiting() > 0:
    reply = messenger.readline()
    reply = reply.decode("utf-8", "ignore")
    print(f"[Pico message]: {reply}")
sleep(2)
messenger.write(
    f"{modes[3]},{pulsewidths[2]},{pulsewidths[1]}\n".encode("utf-8")
)  # recording
if messenger.inWaiting() > 0:
    reply = messenger.readline()
    reply = reply.decode("utf-8", "ignore")
    print(f"[Pico message]: {reply}")
sleep(2)
messenger.write(
    f"{modes[4]},{pulsewidths[2]},{pulsewidths[2]}\n".encode("utf-8")
)  # autopilot
if messenger.inWaiting() > 0:
    reply = messenger.readline()
    reply = reply.decode("utf-8", "ignore")
    print(f"[Pico message]: {reply}")
sleep(2)
messenger.write(
    f"{modes[5]},{pulsewidths[0]},{pulsewidths[0]}\n".encode("utf-8")
)  # error
if messenger.inWaiting() > 0:
    reply = messenger.readline()
    reply = reply.decode("utf-8", "ignore")
    print(f"[Pico message]: {reply}")
sleep(1)
messenger.close()
