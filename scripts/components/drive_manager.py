import sys
import pygame


class DriveManager:
    def __init__(self, joy_id=0, autopilot_on=False) -> None:
        # pygame.display.init()
        pygame.init()
        pygame.joystick.init()
        if pygame.joystick.get_count() == 0:
            print("Error: No joystick connected!")
            sys.exit(1)
        self.gamepad = pygame.joystick.Joystick(joy_id)
        self.gamepad.init()
        print(f"Gamepad initiated at: {self.gamepad.get_name()}\n")
        # Flags
        self.is_terminated = False
        self.is_paused = True
        self.is_recording = False
        self.autopilot_on = autopilot_on
        # Variables
        self.mode = "p"
        self.steering_value = 0.0
        self.throttle_value = 0.0
        self.steering_pulse_width = 1_500_000
        self.throttle_pulse_width = 1_500_000

    def process_events(self, params):
        for e in pygame.event.get():  # read controller input
            if e.type == pygame.JOYBUTTONDOWN:
                if self.gamepad.get_button(params["terminate_btn"]):  # emergency stop
                    self.is_terminated = True
                    print("E-STOP PRESSED. TERMINATE")
                    pygame.quit()
                    sys.exit()
                elif self.gamepad.get_button(params["pause_btn"]):
                    self.is_paused = not self.is_paused
                    if self.is_paused:
                        self.mode = "p"
                        self.is_recording = False
                    else:
                        if self.autopilot_on:
                            self.mode = "a"
                        else:
                            self.mode = "n"
                    print(f"Paused: {self.is_paused}")
                elif self.gamepad.get_button(params["record_btn"]):
                    if not self.autopilot_on:
                        if not self.is_paused:
                            self.is_recording = not self.is_recording
                            if self.is_recording:
                                self.mode = "r"
                            else:
                                self.mode = "n"
                            print(f"Recording: {self.is_recording}")  # debug
            elif e.type == pygame.JOYAXISMOTION:
                st_ax_val = self.gamepad.get_axis(params["steering_axis"])
                th_ax_val = self.gamepad.get_axis(params["throttle_axis"])
                # Calaculate steering and throttle value
                self.steering_value = round(st_ax_val, 2)  # -1: left end; +1: right end
                self.throttle_value = -round(
                    th_ax_val, 2
                )  # -1:max forward, +1:max reverse
                # Map steering value into pulse width in nanoseconds
                self.steering_pulse_width = params["steering_center"] + int(
                    params["steering_range"] * self.steering_value
                )
                # Map throttle value into pulse width in nanoseconds
                if self.throttle_value > 0:
                    self.throttle_pulse_width = params["throttle_neutral"] + int(
                        params["throttle_fwd_range"]
                        * min(self.throttle_value, params["throttle_limit"])
                    )
                elif self.throttle_value < 0:
                    self.throttle_pulse_width = params["throttle_neutral"] + int(
                        params["throttle_fwd_range"]
                        * max(self.throttle_value, -params["throttle_limit"])
                    )
                else:
                    self.throttle_pulse_width = params["throttle_neutral"]

        return self.mode, self.steering_pulse_width, self.throttle_pulse_width


if __name__ == "__main__":
    import json
    from pathlib import Path
    from time import sleep

    # SETUP
    params_file_path = str(Path(__file__).parents[1].joinpath("configs.json"))
    with open(params_file_path, "r") as file:
        params = json.load(file)
    mngr = DriveManager(joy_id=0, autopilot_on=False)
    print(f"{pygame.joystick.get_count()} joystick connected")

    # LOOP
    try:
        while mngr:
            mode, st_pw, th_pw = mngr.process_events(params)
            print("---")
            print(f"terminate flag: {mngr.is_terminated}")
            print(f"pause flag: {mngr.is_paused}")
            print(f"record flag: {mngr.is_recording}")
            print(f"mode: {mode}")
            print(f"steering value: {mngr.steering_value}, pw: {st_pw}")
            print(f"throttle_value: {mngr.throttle_value}, pw: {th_pw}")
            sleep(0.033)  # 30 Hz
    except KeyboardInterrupt:
        print("\nExiting cleanly...")
    finally:
        pygame.quit()
        sys.exit()
