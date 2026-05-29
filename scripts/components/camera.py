import threading
import queue
from time import sleep
from picamera2 import Picamera2


class ThreadedCamera:
    def __init__(self, params):
        """
        Initializes the camera, configures it, and starts the background thread.
        """
        # Configure the camera
        self.picam = Picamera2()
        config = self.picam.create_video_configuration(
            main={"format": "RGB888", "size": (224, 224)},
            controls={
                "FrameDurationLimits": (
                    int(1_000_000 / params["frame_rate"]),
                    int(1_000_000 / params["frame_rate"]),
                    # int(1_000_000 / 60),
                    # int(1_000_000 / 60),
                )
            },
        )
        self.picam.configure(config)
        # Threading setup
        self.thread_stopped = False  # Flag to cleanly stop the thread
        # Start the camera and the thread
        self.picam.start()
        self.thread = threading.Thread(target=self.refresh_frame, daemon=True)
        self.thread.start()
        # Count down
        for i in reversed(range(3)):
            print(i + 1)
            sleep(1.0)

    def refresh_frame(self):
        """
        The internal thread loop that constantly pulls frames.
        """
        while not self.thread_stopped:
            try:
                self.frame = self.picam.capture_array()
            except Exception as e:
                print(f"Camera thread error: {e}")
                break

    def read(self):
        return self.frame  # block until a frame is available

    def stop(self):
        self.thread_stopped = True
        self.thread.join()  # Wait for the thread to finish
        self.picam.stop()
        print("Camera gracefully stopped.")


if __name__ == "__main__":
    import cv2 as cv
    from pathlib import Path
    import json
    from time import time

    # SETUP
    params_file_path = Path(__file__).parent.joinpath("configs.json")
    with open(params_file_path, "r") as file:
        params = json.load(file)
    print("Initializing camera...")
    cam = ThreadedCamera(params=params)
    print("Starting processing loop. Press Ctrl+C to stop.")

    # LOOP
    try:
        frame_counts = 0
        start_time = time()
        while True:
            frame = cam.read()

            frame_counts += 1

            # Print FPS every sec
            if not frame_counts % params["frame_rate"]:
                elapsed = time() - start_time
                fps = params["frame_rate"] / elapsed
                print(f"Processing at {fps:.2f} FPS | Frame shape: {frame.shape}")
                start_time = time()
            cv.imshow("Camera", cv.flip(frame, -1))  # picam mounted upside down
            if cv.waitKey(1) == ord("q"):  # [q]uit
                print("Quit signal received.")
                break
            sleep(1 / params["frame_rate"])
    except KeyboardInterrupt:
        print("\nShutdown signal received.")
    finally:
        cam.stop()  # always cleanup camera
