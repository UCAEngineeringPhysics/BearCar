import pygame
from time import sleep


# SETUP
print("Please take down the stop button, recording button, steer axis, throttle axis")
sleep(1)

pygame.init()
pygame.joystick.init()
print(f"{pygame.joystick.get_count()} gamepad connected")
gamepad = pygame.joystick.Joystick(0)
print(
    f"Gamepad ID: {gamepad.get_id()}, Name: {gamepad.get_name()}, Button #: {gamepad.get_numbuttons()}, Axis #: {gamepad.get_numaxes()}"
)

# LOOP
while True:
    for e in pygame.event.get():
        if e.type == pygame.JOYAXISMOTION:
            ax0 = gamepad.get_axis(0)
            ax1 = gamepad.get_axis(1)
            ax2 = gamepad.get_axis(2)
            ax3 = gamepad.get_axis(3)
            ax4 = gamepad.get_axis(4)
            ax5 = gamepad.get_axis(5)
            print("---")
            print(f"axis 0: {ax0}")
            print(f"axis 1: {ax1}")
            print(f"axis 2: {ax2}")
            print(f"axis 3: {ax3}")
            print(f"axis 4: {ax4}")
            print(f"axis 5: {ax5}")
            print("---")
        elif e.type == pygame.JOYBUTTONDOWN:
            bt0 = gamepad.get_button(0)
            bt1 = gamepad.get_button(1)
            bt2 = gamepad.get_button(2)
            bt3 = gamepad.get_button(3)
            bt4 = gamepad.get_button(4)
            bt5 = gamepad.get_button(5)
            bt6 = gamepad.get_button(6)
            bt7 = gamepad.get_button(7)
            bt8 = gamepad.get_button(8)
            bt9 = gamepad.get_button(9)
            bt10 = gamepad.get_button(10)
            print("---")
            print(f"button 0: {bt0}")
            print(f"button 1: {bt1}")
            print(f"button 2: {bt2}")
            print(f"button 3: {bt3}")
            print(f"button 4: {bt4}")
            print(f"button 5: {bt5}")
            print(f"button 6: {bt6}")
            print(f"button 7: {bt7}")
            print(f"button 8: {bt8}")
            print(f"button 9: {bt9}")
            print(f"button 10: {bt10}")
            print("---")
