import cv2
import numpy as np
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import gym
import time
import os 

env = gym.make('SuperMarioBros-v0', apply_api_compatibility=True, render_mode="human")
env = JoypadSpace(env, SIMPLE_MOVEMENT)

# -----------------------------------
# ACTIONS
# -----------------------------------
RIGHT = 1
RIGHT_JUMP = 2
CUSTOM_JUMP = [4] * 23
UP_JUMP = [5] * 23
JUMP = 5
# ------------------------------------
# Bounds for items
# ------------------------------------

lower_bound_goomba = np.array([12, 0, 0])
upper_bound_goomba = np.array([15, 175, 255])

lower_bound_pipe = np.array([23, 255, 0])
upper_bound_pipe = np.array([179, 255, 255])

lower_bound_koopa = np.array([0, 151, 243])
upper_bound_koopa = np.array([179, 187, 255])

# ------------------------------------
# Global variables
# ------------------------------------
last_x_position = None
obstacle_detected = False  
no_ground_detected = False  


# ------------------------------------
# Functions
# ------------------------------------

def adjust_region_of_interest(frame, region_start_percentage, region_end_percentage, vertical_start_percentage, vertical_end_percentage):
    return frame[int(frame.shape[0] * vertical_start_percentage):int(frame.shape[0] * vertical_end_percentage),
                   int(frame.shape[1] * region_start_percentage):int(frame.shape[1] * region_end_percentage)]

def detect_goomba(frame):
    # How to detect a goomba:
    # 1. Adjust the region of interest
    roi = adjust_region_of_interest(frame, 0.5, 0.64, 0.7, 0.87)
    hsv_frame = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
    mask_goomba = cv2.inRange(hsv_frame, lower_bound_goomba, upper_bound_goomba)
    num_pixels = np.sum(mask_goomba)

    # 5. If the number of pixels is greater than 100, then a goomba is detected
    if num_pixels > 1000:
        print("Goomba detected")
        return True

    return False

def detect_pipe(frame):
    # How to detect a pipe:
    # 1. Adjust the region of interest
    roi = adjust_region_of_interest(frame, 0.4, 0.8, 0.7, 0.75)
    hsv_frame = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
    mask_pipe = cv2.inRange(hsv_frame, lower_bound_pipe, upper_bound_pipe)
    num_pixels = np.sum(mask_pipe)

    # 5. If the number of pixels is greater than 100, then a pipe is detected
    if num_pixels > 1000:
        return True

    return False

def visualize_hsv_bounds(frame):
    roi = adjust_region_of_interest(frame, 0.6, 0.9, 0.7, 0.87)
    hsv_frame = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)

    # Create masks for different bounds
    mask_goomba = cv2.inRange(hsv_frame, lower_bound_goomba, upper_bound_goomba)
    mask_pipe = cv2.inRange(hsv_frame, lower_bound_pipe, upper_bound_pipe)
    mask_koopa = cv2.inRange(hsv_frame, lower_bound_koopa, upper_bound_koopa)

    # Combine the masks
    mask_combined = cv2.bitwise_or(cv2.bitwise_or(mask_goomba, mask_pipe), mask_koopa)

    # Display the masks
    cv2.imshow("Goomba Mask", mask_goomba)
    cv2.imshow("Pipe Mask", mask_pipe)
    cv2.imshow("Koopa Mask", mask_koopa)
    cv2.imshow("Combined Mask", mask_combined)

    # Wait for a key press to close the windows
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def detect_ground(frame):
    sky_row = 10
    sky_color = frame[sky_row, frame.shape[1] // 2]  

    ground_row = int(frame.shape[0] * 0.9)  
    ground_color = frame[ground_row, frame.shape[1] // 2 + 15]

    is_missing_ground = np.array_equal(sky_color, ground_color)

    return is_missing_ground

def detect_koopa(frame):
    roi = adjust_region_of_interest(frame, 0.6, 0.9, 0.7, 0.87)
    hsv_frame = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
    mask_koopa = cv2.inRange(hsv_frame, lower_bound_koopa, upper_bound_koopa)
    num_pixels = np.sum(mask_koopa)

    # 5. If the number of pixels is greater than 100, then a goomba is detected
    if num_pixels > 1000:
        return True

    return False

done = True
env.reset()
obs, reward, terminated, truncated, info = env.step(RIGHT)
mario_x_position = info["x_pos"]
action_index = RIGHT
prev_x_position = mario_x_position


for step in range(10000):
    obs, reward, terminated, truncated, info = env.step(action_index)
    mario_x_position = info["x_pos"]
    no_ground_detected = detect_ground(obs)
    goomba_detected = detect_goomba(obs)
    pipe_detected = detect_pipe(obs)
    koopa_detected = detect_koopa(obs)

    if goomba_detected:
        for jump_frame in CUSTOM_JUMP:
            obs, reward, terminated, truncated, info = env.step(RIGHT_JUMP)
            print("Goomba detected")

    elif pipe_detected:
        if mario_x_position > prev_x_position:
            for jump_frame in UP_JUMP:
                obs, reward, terminated, truncated, info = env.step(JUMP)
                print("Pipe detected")
            obs, reward, terminated, truncated, info = env.step(RIGHT)
        else:
            for jump_frame in UP_JUMP:
                obs, reward, terminated, truncated, info = env.step(RIGHT_JUMP)


    elif no_ground_detected:
        for jump_frame in UP_JUMP:
            obs, reward, terminated, truncated, info = env.step(RIGHT_JUMP)
            print("No ground detected")

    elif koopa_detected:
        for jump_frame in UP_JUMP:
            obs, reward, terminated, truncated, info = env.step(JUMP)
            print("Koopa detected")

    else:
        obs, reward, terminated, truncated, info = env.step(RIGHT)

    # Visualize the HSV bounds on the frame
    if step % 10 == 0:
        visualize_hsv_bounds(obs)

    # Update the previous x position
    prev_x_position = mario_x_position

    done = terminated or truncated
    if done:
        env.reset()

env.close()
