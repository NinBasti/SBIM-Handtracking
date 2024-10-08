import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import time
import threading
from pynput import keyboard

# Initialize MediaPipe hands module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.35,
    min_tracking_confidence=0.35,
    model_complexity=0
)

# Set up the webcam
cap = cv2.VideoCapture(1)
ret, frame = cap.read()
if not ret:
    print("Failed to capture video")
    exit(1)

# Configure PyAutoGUI
pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0

# Get screen size
screen_width, screen_height = pyautogui.size()

# Define the portion of the camera view to map to the full screen (70% here)
inner_area_percent = 0.7

# Calculate the margins around the inner area
def calculate_margins(frame_width, frame_height, inner_area_percent):
    margin_width = frame_width * (1 - inner_area_percent) / 2
    margin_height = frame_height * (1 - inner_area_percent) / 2
    return margin_width, margin_height

# Convert video coordinates to screen coordinates
def convert_to_screen_coordinates(x, y, frame_width, frame_height, margin_width, margin_height):
    screen_x = np.interp(x, (margin_width, frame_width - margin_width), (0, screen_width))
    screen_y = np.interp(y, (margin_height, frame_height - margin_height), (0, screen_height))
    return screen_x, screen_y

# Movement Thread for smoother cursor movement
class CursorMovementThread(threading.Thread):
    def __init__(self):
        super().__init__()
        self.daemon = True
        self.current_x, self.current_y = pyautogui.position()
        self.target_x, self.target_y = self.current_x, self.current_y
        self.running = True
        self.active = False  # Mouse movement initially inactive
        self.jitter_threshold = 0.003
        self.smooth_transition_speed = 0.2

    def run(self):
        while self.running:
            if self.active:
                distance = np.hypot(self.target_x - self.current_x, self.target_y - self.current_y)
                screen_diagonal = np.hypot(screen_width, screen_height)
                if distance / screen_diagonal > self.jitter_threshold:
                    step = max(0.0001, distance * self.smooth_transition_speed)  # Smooth transition speed
                    if distance != 0:
                        step_x = (self.target_x - self.current_x) / distance * step
                        step_y = (self.target_y - self.current_y) / distance * step
                        self.current_x += step_x
                        self.current_y += step_y
                        pyautogui.moveTo(self.current_x, self.current_y, _pause=False)
                time.sleep(0)
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

# Initialize the movement thread
movement_thread = CursorMovementThread()
movement_thread.start()

# Variable to track whether left click is toggled on or off
left_click_enabled = False

# Variable to track whether mouse movement is enabled or disabled
mouse_movement_enabled = True  # Tracks whether the user has enabled/disabled movement
tracking_active = True  # Tracks the state of tracking regardless of hand detection
tracking_lost = False  # Tracks if tracking was lost

# Function to draw hand landmarks and connections
def draw_landmarks(frame, hand_landmarks):
    if hand_landmarks:
        for landmark in hand_landmarks.landmark:
            h, w, _ = frame.shape
            cx, cy = int(landmark.x * w), int(landmark.y * h)
            cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)  # Draw landmark
        # Draw connections between landmarks
        mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

# Thread to handle left click toggle
def handle_left_click():
    global left_click_enabled
    while True:
        if left_click_enabled:
            pyautogui.mouseDown()
        else:
            pyautogui.mouseUp()
        time.sleep(0.1)  # Small delay to avoid too frequent actions

# Start the left click thread
click_thread = threading.Thread(target=handle_left_click)
click_thread.daemon = True
click_thread.start()

# Toggle function for the left click
def toggle_left_click():
    global left_click_enabled
    left_click_enabled = not left_click_enabled
    print(f"Left click {'enabled' if left_click_enabled else 'disabled'}")

# Toggle function for mouse movement
def toggle_mouse_movement():
    global mouse_movement_enabled, tracking_active
    mouse_movement_enabled = not mouse_movement_enabled
    print(f"Mouse movement {'enabled' if mouse_movement_enabled else 'disabled'}")

    # Adjust based on the toggle, without double-toggling
    if not mouse_movement_enabled:
        movement_thread.deactivate()
    elif tracking_active:  # Reactivate only if tracking is active
        movement_thread.activate()

# Function to handle key press events
def on_press(key):
    try:
        if key.char == 'c':
            toggle_left_click()
        elif key.char == 't':
            toggle_mouse_movement()
    except AttributeError:
        pass

# Set up the listener for keyboard events
listener = keyboard.Listener(on_press=on_press)
listener.start()

try:
    while True:
        # Read a frame from the webcam
        ret, frame = cap.read()
        if not ret:
            continue

        # Flip the frame horizontally for a natural selfie-view, and convert the BGR image to RGB
        frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)

        # Process the frame and find hands
        results = hands.process(frame)

        # Convert the frame color back so it can be displayed
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Check for the presence of hands
        if results.multi_hand_landmarks:
            tracking_active = True  # Hand tracking is detected
            tracking_lost = False  # Reset tracking loss
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks and connections
                draw_landmarks(frame, hand_landmarks)

                # Use the base of the ring finger (RING_FINGER_MCP) for tracking
                ring_finger_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP]
                mcp_x = int(ring_finger_mcp.x * frame.shape[1])
                mcp_y = int(ring_finger_mcp.y * frame.shape[0])

                # Calculate margins based on the current frame size
                margin_width, margin_height = calculate_margins(frame.shape[1], frame.shape[0], inner_area_percent)

                # Convert video coordinates to screen coordinates
                target_x, target_y = convert_to_screen_coordinates(mcp_x, mcp_y, frame.shape[1], frame.shape[0],
                                                                   margin_width, margin_height)

                # Update target position in movement thread if mouse movement is enabled
                if mouse_movement_enabled:
                    movement_thread.activate()  # Make sure movement is active
                    movement_thread.update_target(target_x, target_y)
        else:
            if not tracking_lost:
                tracking_lost = True  # Mark that we lost tracking
            tracking_active = False  # No hand detected
            if mouse_movement_enabled:  # Only deactivate if movement is supposed to be enabled
                movement_thread.deactivate()

        # Display the frame with landmarks
        cv2.imshow('Hand Tracking', frame)

        if cv2.waitKey(1) & 0xFF == 27:  # Press 'ESC' to exit
            break

finally:
    movement_thread.stop()
    cap.release()
    cv2.destroyAllWindows()
