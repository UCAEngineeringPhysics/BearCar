"""
This script will test communication between Pico and RPi.
Please uncomment line 44 and 79 in the 'main.py' on Pico.
"""

from serial import Serial
from time import sleep

# SAFETY CHECK
is_lifted = input("Is something contacting any wheel of BearCart? (Y/n)")
while is_lifted != 'n':
    print("Please lift BearCart up and remove everything that is making the contact")
    is_lifted = input("Is something contacting any wheel of BearCart? (Y/n)")
print("Hold tight! You are about to unleash the beast!")
for i in reversed(range(3)):
    print(i+1)
    sleep(1)

# SETUP
messenger = Serial(port='/dev/ttyACM0', baudrate=115200)
print(messenger.name)

# LOOP
# msg = "n, 1_876_543, 1_543_210\n".encode('utf-8')
msg = "Hello from RPi\n".encode('utf-8')
for i in range(100):
    messenger.write(f"Hello from RPi: {i}\n".encode('utf-8'))
    # messenger.write(msg)
    # print("Transmitting")  # debug
    sleep(0.1)
    if messenger.inWaiting() > 0:
        reply = messenger.readline()
        reply = reply.decode('utf-8', 'ignore')
        print(f"[Pico Response] {reply}")
messenger.close()
