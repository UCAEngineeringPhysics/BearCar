import sys
import threading
import pygame
from time import sleep


class Controller:
    def __init__(
        self,
        joy_id=0,
        terminate_btn_id=0,
        pause_btn_id=4,
        record_btn_id=5,
        steering_axis_id=0,
        throttle_axis_id=4,
    ) -> None:
        pygame.display.init()
        pygame.joystick.init()
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
        # Start thread to process gamepad input
        self.gamepad_thread = threading.Thread(target=self.process_events, daemon=True)
        self.gamepad_thread.start()

    def process_events(self):
        while self.gamepad:
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
                    steering_axval = self.gamepad.get_axis(self.steering_axis_id)
                    throttle_axval = self.gamepad.get_axis(self.throttle_axis_id)
            # Calaculate steering and throttle value
            self.steering_value = round(
                steering_axval, 3
            )  # -1: left end; +1: right end
            self.throttle_value = -round(
                throttle_axval, 3
            )  # -1: max forward, +1: max reverse


if __name__ == "__main__":
    # SETUP
    con = Controller()
    print(f"{pygame.joystick.get_count()} joystick connected")

    # LOOP
    while True:
        print("---")
        print(f"terminate flag: {con.is_terminated}")
        print(f"pause flag: {con.is_paused}")
        print(f"record flag: {con.is_recording}")
        print(f"steering value: {con.steering_value}")
        print(f"throttle_value: {con.throttle_value}")
        sleep(0.1)
