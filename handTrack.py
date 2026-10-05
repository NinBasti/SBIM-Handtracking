import os
import time
import threading
import urllib.request
import atexit
import signal

import cv2
import mediapipe as mp
import numpy as np
import pyautogui
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

pyautogui_lock = threading.Lock()

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/"
             "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task")
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")

if not os.path.exists(MODEL_PATH):
    print("Downloading hand_landmarker.task ...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

options = vision.HandLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=vision.RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.35,
    min_hand_presence_confidence=0.35,
    min_tracking_confidence=0.35,
)
landmarker = vision.HandLandmarker.create_from_options(options)

RING_FINGER_MCP = 13

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index
    (5, 9), (9, 10), (10, 11), (11, 12),     # middle
    (9, 13), (13, 14), (14, 15), (15, 16),   # ring
    (13, 17), (17, 18), (18, 19), (19, 20),  # pinky
    (0, 17),                                 # palm edge
]

cap = cv2.VideoCapture(0)
ret, frame = cap.read()
if not ret:
    print("Failed to capture video")
    exit(1)

pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0

screen_width, screen_height = pyautogui.size()
inner_area_percent = 0.7


def calculate_margins(frame_width, frame_height, inner_area_percent):
    margin_width = frame_width * (1 - inner_area_percent) / 2
    margin_height = frame_height * (1 - inner_area_percent) / 2
    return margin_width, margin_height


def convert_to_screen_coordinates(x, y, frame_width, frame_height, margin_width, margin_height):
    screen_x = np.interp(x, (margin_width, frame_width - margin_width), (0, screen_width))
    screen_y = np.interp(y, (margin_height, frame_height - margin_height), (0, screen_height))
    return screen_x, screen_y


def release_mouse():
    with pyautogui_lock:
        pyautogui.mouseUp()


atexit.register(release_mouse)


# ---------------------------------------------------------------------------
# Cursor movement thread
# ---------------------------------------------------------------------------
class CursorMovementThread(threading.Thread):
    def __init__(self):
        super().__init__()
        self.daemon = True
        self.current_x, self.current_y = pyautogui.position()
        self.target_x, self.target_y = self.current_x, self.current_y
        self.running = True
        self.active = False
        self.jitter_threshold = 0.003
        self.smooth_transition_speed = 0.2

    def run(self):
        while self.running:
            if self.active:
                distance = np.hypot(self.target_x - self.current_x, self.target_y - self.current_y)
                screen_diagonal = np.hypot(screen_width, screen_height)
                if distance / screen_diagonal > self.jitter_threshold:
                    step = max(0.0001, distance * self.smooth_transition_speed)
                    if distance != 0:
                        step_x = (self.target_x - self.current_x) / distance * step
                        step_y = (self.target_y - self.current_y) / distance * step
                        self.current_x += step_x
                        self.current_y += step_y
                        with pyautogui_lock:
                            pyautogui.moveTo(self.current_x, self.current_y, _pause=False)
                time.sleep(0.01)
            else:
                time.sleep(0.1)

    def update_target(self, x, y):
        self.target_x, self.target_y = x, y

    def activate(self):
        self.active = True

    def deactivate(self):
        self.active = False

    def stop(self):
        self.running = False


movement_thread = CursorMovementThread()
movement_thread.start()

left_click_enabled = False
mouse_movement_enabled = True
tracking_active = True
tracking_lost = False


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------
def draw_landmarks(frame, landmarks):
    h, w, _ = frame.shape
    points = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, points[a], points[b], (255, 255, 255), 2)
    for p in points:
        cv2.circle(frame, p, 5, (0, 255, 0), -1)


# ---------------------------------------------------------------------------
# Click handling
# ---------------------------------------------------------------------------
def handle_left_click():
    global left_click_enabled
    is_pressed = False
    while True:
        if left_click_enabled and not is_pressed:
            with pyautogui_lock:
                pyautogui.mouseDown()
            is_pressed = True
        elif not left_click_enabled and is_pressed:
            with pyautogui_lock:
                pyautogui.mouseUp()
            is_pressed = False
        time.sleep(0.05)


click_thread = threading.Thread(target=handle_left_click)
click_thread.daemon = True
click_thread.start()


def toggle_left_click():
    global left_click_enabled
    left_click_enabled = not left_click_enabled
    if not left_click_enabled:
        with pyautogui_lock:
            pyautogui.mouseUp()
    print(f"Left click {'enabled' if left_click_enabled else 'disabled'}")


def toggle_mouse_movement():
    global mouse_movement_enabled, tracking_active
    mouse_movement_enabled = not mouse_movement_enabled
    print(f"Mouse movement {'enabled' if mouse_movement_enabled else 'disabled'}")

    if not mouse_movement_enabled:
        movement_thread.deactivate()
    elif tracking_active:
        movement_thread.activate()


# ---------------------------------------------------------------------------
#   SIGUSR1 -> toggle left click
#   SIGUSR2 -> toggle mouse movement
# ---------------------------------------------------------------------------
signal.signal(signal.SIGUSR1, lambda *_: toggle_left_click())
signal.signal(signal.SIGUSR2, lambda *_: toggle_mouse_movement())

print(f"Running (PID {os.getpid()}). Toggle click: pkill -USR1 -f '[h]and_mouse.py' | "
      f"Toggle movement: pkill -USR2 -f '[h]and_mouse.py'")

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
last_timestamp_ms = -1

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int(time.monotonic() * 1000)
        if timestamp_ms <= last_timestamp_ms:
            timestamp_ms = last_timestamp_ms + 1
        last_timestamp_ms = timestamp_ms

        results = landmarker.detect_for_video(mp_image, timestamp_ms)

        if results.hand_landmarks:
            tracking_active = True
            tracking_lost = False
            for hand_landmarks in results.hand_landmarks:
                draw_landmarks(frame, hand_landmarks)

                ring_finger_mcp = hand_landmarks[RING_FINGER_MCP]
                mcp_x = int(ring_finger_mcp.x * frame.shape[1])
                mcp_y = int(ring_finger_mcp.y * frame.shape[0])

                margin_width, margin_height = calculate_margins(
                    frame.shape[1], frame.shape[0], inner_area_percent)

                target_x, target_y = convert_to_screen_coordinates(
                    mcp_x, mcp_y, frame.shape[1], frame.shape[0],
                    margin_width, margin_height)

                if mouse_movement_enabled:
                    movement_thread.activate()
                    movement_thread.update_target(target_x, target_y)
        else:
            if not tracking_lost:
                tracking_lost = True
            tracking_active = False
            if mouse_movement_enabled:
                movement_thread.deactivate()

        cv2.imshow('Hand Tracking', frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

finally:
    release_mouse()
    movement_thread.stop()
    landmarker.close()
    cap.release()
    cv2.destroyAllWindows()