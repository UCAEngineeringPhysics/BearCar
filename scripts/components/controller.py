import sys
import pygame
from time import sleep


class Controller:
    def __init__(
        self,
        terminate_btn_id,
        pause_btn_id,
        record_btn_id,
        steering_axis_id,
        throttle_axis_id,
        joy_id=0,
    ) -> None:
        # pygame.display.init()
        pygame.init()
        pygame.joystick.init()
        if pygame.joystick.get_count() == 0:
            print("Error: No joystick connected!")
            sys.exit(1)
        self.gamepad = pygame.joystick.Joystick(joy_id)
        print(f"Gamepad initiated at: {self.gamepad.get_name()}\n")
        # Buttons and Axes id #
        self.terminate_btn_id = terminate_btn_id
        self.pause_btn_id = pause_btn_id
        self.record_btn_id = record_btn_id
        self.steering_axis_id = steering_axis_id
        self.throttle_axis_id = throttle_axis_id
        # Flags
        self.is_terminated = False
        self.is_paused = True
        self.is_recording = False
        # Variables
        self.steering_value = 0.0
        self.throttle_value = 0.0

    def process_events(self):
        for e in pygame.event.get():  # read controller input
            if e.type == pygame.JOYBUTTONDOWN:
                if self.gamepad.get_button(self.terminate_btn_id):  # emergency stop
                    self.is_terminated = True
                    print("E-STOP PRESSED. TERMINATE")
                    # pygame.quit()
                    # sys.exit()
                elif self.gamepad.get_button(self.pause_btn_id):
                    self.is_paused = not self.is_paused
                    if self.is_paused:
                        self.is_recording = False
                    print(f"Paused: {self.is_paused}")
                elif self.gamepad.get_button(self.record_btn_id):
                    if not self.is_paused:
                        self.is_recording = not self.is_recording
                        print(f"Recording: {self.is_recording}")  # debug
            elif e.type == pygame.JOYAXISMOTION:
                st_ax_val = self.gamepad.get_axis(self.steering_axis_id)
                th_ax_val = self.gamepad.get_axis(self.throttle_axis_id)
                # Calaculate steering and throttle value
                self.steering_value = round(st_ax_val, 3)  # -1: left end; +1: right end
                self.throttle_value = -round(
                    th_ax_val, 3
                )  # -1: max forward, +1: max reverse


if __name__ == "__main__":
    # SETUP
    con = Controller(
        terminate_btn_id=0,
        pause_btn_id=4,
        record_btn_id=5,
        steering_axis_id=0,
        throttle_axis_id=4,
        joy_id=0,
    )
    print(f"{pygame.joystick.get_count()} joystick connected")

    # LOOP
    try:
        while con:
            con.process_events()
            print("---")
            print(f"terminate flag: {con.is_terminated}")
            print(f"pause flag: {con.is_paused}")
            print(f"record flag: {con.is_recording}")
            print(f"steering value: {con.steering_value}")
            print(f"throttle_value: {con.throttle_value}")
            sleep(0.033)  # 30 Hz
    except KeyboardInterrupt:
        print("\nExiting cleanly...")
    finally:
        pygame.quit()
        sys.exit()
