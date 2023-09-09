import cv2
import numpy as np
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import matplotlib.pyplot as plt
import gym
import time

# Initialize environment
env = gym.make('SuperMarioBros-v0', apply_api_compatibility=True, render_mode="human")
env = JoypadSpace(env, SIMPLE_MOVEMENT)

# Define actions
RIGHT = 1
RIGHT_JUMP = 2
RIGHT_SPEED_JUMP = 4

# Custom jump action (hold 'A' button for longer)
#This is hard coded atm (change number that is * to see speed)
CUSTOM_JUMP = [4] * 23 # Hold 'A' button for 27 frames

# Goomba Color Range
lower_bound_goomba = np.array([0, 237, 227])
upper_bound_goomba = np.array([179, 255, 247])

# Pipe Color Ranges (using the values you provided)
lower_bound_pipe = np.array([38, 0, 0])
upper_bound_pipe = np.array([41, 230, 255])

# Ground Color Range (you may need to adjust these values)
lower_bound_ground = np.array([0, 237, 227])
upper_bound_ground = np.array([179, 255, 247])

INITIAL_DELAY = 10  # Added constant for initial delay

last_x_position = None
obstacle_detected = False  # Flag to keep track of obstacle detection
no_ground_detected = False  # Flag to keep track of no ground detection

def detect_obstacle(frame, mario_x_position):
    global last_x_position, obstacle_detected
    # Adjusting the region of interest
    region_start_percentage = 0.3
    region_end_percentage = 0.7
    vertical_start_percentage = 0.5  # Adjusted to avoid sky boxes
    vertical_end_percentage = 0.9

    region = frame[int(frame.shape[0] * vertical_start_percentage):int(frame.shape[0] * vertical_end_percentage),
                   int(frame.shape[1] * region_start_percentage):int(frame.shape[1] * region_end_percentage)]

    # Display the region
    cv2.imshow('Region of Interest', region)
    cv2.waitKey(1)

    hsv_region = cv2.cvtColor(region, cv2.COLOR_RGB2HSV)

    # Goomba Mask
    mask_goomba = cv2.inRange(hsv_region, lower_bound_goomba, upper_bound_goomba)

    # Pipe Masks
    mask_pipe = cv2.inRange(hsv_region, lower_bound_pipe, upper_bound_pipe)

    # Combine Goomba and Pipe masks
    mask_combined = cv2.bitwise_or(mask_goomba, mask_pipe)

    # Debugging - Display the combined mask
    cv2.imshow('Obstacle Detection Mask', mask_combined)
    cv2.waitKey(1)

    # Check for contours to differentiate between Goomba and Pipe
    contours, _ = cv2.findContours(mask_goomba, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    obstacles_detected = []

    for contour in contours:
        if cv2.contourArea(contour) > 50:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w/h

            obstacles_detected.append((aspect_ratio, h))

            # Check aspect ratio for Goomba
            if 1.6 <= aspect_ratio <= 1.8:  # Slightly broadened for safety
                return "goomba"

            # Check if the aspect ratio is 1.88 (pipe)
            if aspect_ratio == 1.88 and not obstacle_detected:
                obstacle_detected = True
                return "pipe"

    # Reset the obstacle detection flag when Mario moves
    if mario_x_position != last_x_position:
        obstacle_detected = False

    return obstacles_detected

#this needs fixing 
#will check colour of absent ground and jump before with mask
def detect_ground(frame):
    # 1. Identify the Sky Color:
    sky_row = 10  # Y-coordinate for a row that's always in the sky
    sky_color = frame[sky_row, frame.shape[1] // 2]  # Color in the middle of that row

    # 2. Detect Missing Ground:
    ground_row = int(frame.shape[0] * 0.9)  # Y-coordinate near the bottom (90% of the frame's height)
    ground_color = frame[ground_row, frame.shape[1] // 2 + 15]  # Color in the middle of that row

    is_missing_ground = np.array_equal(sky_color, ground_color)
    
    return is_missing_ground

done = True
env.reset()

action_index = RIGHT
for step in range(5000):
    obs, reward, terminated, truncated, info = env.step(action_index)
    
    mario_x_position = info["x_pos"]
    
    # Check for ground detection every frame
    ground_detected = detect_ground(obs)
    if ground_detected:
        for jump_frame in CUSTOM_JUMP:
            obs, reward, terminated, truncated, info = env.step(RIGHT_JUMP)
        no_ground_detected = True
    else:
        no_ground_detected = False
    
    if last_x_position is not None and last_x_position == mario_x_position:
        stopped_frames += 1

        # If Mario hasn't moved for more than 10 frames, analyze obstacle
        if stopped_frames >= 10:
            obstacles_detected = detect_obstacle(obs, mario_x_position)
            if isinstance(obstacles_detected, list):
                for aspect_ratio, height in obstacles_detected:
                    if aspect_ratio == 1.88:  # Check if "pipe" is detected
                        # Perform a custom jump
                        for jump_frame in CUSTOM_JUMP:
                            obs, reward, terminated, truncated, info = env.step(RIGHT_JUMP)
                    print(f"Obstacle detected when Mario stopped! Aspect Ratio: {aspect_ratio:.2f} Height: {height}")

    else:
        stopped_frames = 0

    last_x_position = mario_x_position  # Update the last position
    
    obstacle = detect_obstacle(obs, mario_x_position)
    if obstacle == "goomba" or obstacle == "pipe":
        # Perform a custom jump
        for jump_frame in CUSTOM_JUMP:
            obs, reward, terminated, truncated, info = env.step(RIGHT_JUMP)
    else:
        # Use a regular action
        obs, reward, terminated, truncated, info = env.step(RIGHT)

    done = terminated or truncated
    if done:
        env.reset()

    time.sleep(0.01)

env.close()
