import sys
from pathlib import Path
import pygame
import torch


class Driver:
    """
    Manages human's or autopilot's input for BearCar
    """

    def __init__(self, joy_id=0, autopilot_model=None) -> None:
        # Init gamepad
        pygame.init()
        pygame.joystick.init()
        if pygame.joystick.get_count() == 0:
            print("Error: No joystick connected!")
            sys.exit(1)
        self.gamepad = pygame.joystick.Joystick(joy_id)
        self.gamepad.init()
        print(f"Gamepad initiated at: {self.gamepad.get_name()}\n")
        # Init autopilot, if available
        self.autopilot = autopilot_model  # default to human driver
        # if autopilot_name:  # None for human
        #     self.to_tensor = v2.Compose(
        #         [v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]
        #     )
        #     self.autopilot = BearNet()
        #     model_path = Path(__file__).parents[2].joinpath("models", autopilot_name)
        #     self.autopilot.load_state_dict(
        #         torch.load(
        #             model_path,
        #             weights_only=True,
        #             map_location=torch.device("cpu"),
        #         )
        #     )
        #     self.autopilot.eval()  # freeze weights
        # Flags
        self.is_terminated = False
        self.is_paused = True
        self.is_recording = False
        # Variables
        self.mode = "p"  # init mode: pause
        self.steering_value = 0.0
        self.throttle_value = 0.0
        self.steering_pulse_width = 1_500_000  # center
        self.throttle_pulse_width = 1_500_000  # stall

    def process_event(self, params, frame=None):
        for e in pygame.event.get():  # read controller input
            if e.type == pygame.JOYBUTTONDOWN:  # check buttons pressed
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
                    else:  # if not paused
                        if self.autopilot:
                            self.mode = "a"
                        else:
                            self.mode = "n"
                    print(f"Paused: {self.is_paused}")
                elif self.gamepad.get_button(params["record_btn"]):
                    if not self.autopilot:  # only work for human driver
                        if not self.is_paused:  # only work in normal mode
                            self.is_recording = not self.is_recording
                            if self.is_recording:
                                self.mode = "r"
                            else:
                                self.mode = "n"
                            print(f"Recording: {self.is_recording}")
            elif e.type == pygame.JOYAXISMOTION:
                if self.autopilot:
                    img_tensor = self.to_tensor(
                        frame[:, :, [2, 1, 0]]
                    )  # WARN: autopilot needs BGR
                    with torch.no_grad():
                        self.steering_value, self.throttle_value = map(
                            float,
                            torch.clamp(
                                self.autopilot(img_tensor[None, :]).squeeze(),
                                min=-0.999,
                                max=0.999,
                            ),
                        )
                else:  # human input from gamepad
                    st_ax_val = self.gamepad.get_axis(params["steering_axis"])
                    th_ax_val = self.gamepad.get_axis(params["throttle_axis"])
                    # Calaculate steering and throttle value
                    self.steering_value = round(
                        st_ax_val, 2
                    )  # -1: left end; +1: right end
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
    params_file_path = Path(__file__).parent.joinpath("configs.json")
    with open(params_file_path, "r") as file:
        params = json.load(file)
    driver = Driver(joy_id=0, autopilot_model=None)
    print(f"{pygame.joystick.get_count()} joystick connected")

    # LOOP
    try:
        while driver.gamepad:
            mode, st_pw, th_pw = driver.process_event(params, frame=None)
            print("---")
            print(f"terminate flag: {driver.is_terminated}")
            print(f"pause flag: {driver.is_paused}")
            print(f"record flag: {driver.is_recording}")
            print(f"mode: {mode}")
            print(f"steering value: {driver.steering_value}, pw: {st_pw}")
            print(f"throttle_value: {driver.throttle_value}, pw: {th_pw}")
            sleep(0.033)  # 30 Hz
    except KeyboardInterrupt:
        print("\nExiting cleanly...")
    finally:
        pygame.quit()
        sys.exit()
