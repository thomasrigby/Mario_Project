import cv2
import numpy as np
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import gym
import time
import os 

# Initialize environment
env = gym.make('SuperMarioBros-v0', apply_api_compatibility=True, render_mode="human")
env = JoypadSpace(env, SIMPLE_MOVEMENT)

# Define actions
RIGHT = 1
RIGHT_JUMP = 2


# Custom jump action (hold 'A' button for longer)
CUSTOM_JUMP = [4] * 23  # Hold 'A' button for 27 frames

# # # Create a window for trackbars
# cv2.namedWindow('Trackbars')

# # Function to update Koopa color range using trackbars
# def update_koopa_color_range(*args):
#     global lower_bound_koopa, upper_bound_koopa
#     lower_bound_koopa = np.array([
#         cv2.getTrackbarPos('Lower H', 'Trackbars'),
#         cv2.getTrackbarPos('Lower S', 'Trackbars'),
#         cv2.getTrackbarPos('Lower V', 'Trackbars')
#     ])
#     upper_bound_koopa = np.array([
#         cv2.getTrackbarPos('Upper H', 'Trackbars'),
#         cv2.getTrackbarPos('Upper S', 'Trackbars'),
#         cv2.getTrackbarPos('Upper V', 'Trackbars')
#     ])
#     display_koopa_mask()

# def display_koopa_mask():
#     hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#     mask_koopa = cv2.inRange(hsv, lower_bound_koopa, upper_bound_koopa)
#     result = cv2.bitwise_and(frame, frame, mask=mask_koopa)
#     cv2.imshow('Koopa Color Mask', result)

# # Create trackbars for Koopa color range
# cv2.createTrackbar('Lower H', 'Trackbars', 0, 179, update_koopa_color_range)
# cv2.createTrackbar('Lower S', 'Trackbars', 0, 255, update_koopa_color_range)
# cv2.createTrackbar('Lower V', 'Trackbars', 0, 255, update_koopa_color_range)
# cv2.createTrackbar('Upper H', 'Trackbars', 179, 179, update_koopa_color_range)
# cv2.createTrackbar('Upper S', 'Trackbars', 255, 255, update_koopa_color_range)
# cv2.createTrackbar('Upper V', 'Trackbars', 255, 255, update_koopa_color_range)

# # Load the frame from the frames directory
# frame_path = "frames/frame_80.png"
# frame = cv2.imread(frame_path)

# # Check if the frame was loaded correctly
# if frame is not None:
#     # Display the frame and initial mask
#     cv2.imshow('Loaded Frame', frame)
#     display_koopa_mask()
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
# else:
#     print(f"Failed to load the frame from {frame_path}")

# Goomba Color Range
lower_bound_goomba = np.array([14, 0, 0])
upper_bound_goomba = np.array([17, 184, 255])

# Pipe Color Ranges (using the values you provided)
lower_bound_pipe = np.array([12, 196, 0])
upper_bound_pipe = np.array([179, 254, 255])

# Ground Color Range (you may need to adjust these values)
lower_bound_ground = np.array([0, 237, 227])
upper_bound_ground = np.array([179, 255, 247])

# Koopa Color Range (using the values you provided)
lower_bound_koopa = np.array([0, 0, 0])
upper_bound_koopa = np.array([0, 0, 255])



INITIAL_DELAY = 20  # Added constant for initial delay

last_x_position = None
obstacle_detected = False  # Flag to keep track of obstacle detection
no_ground_detected = False  # Flag to keep track of no ground detection

# frame_counter = 0
# frame_interval = 10  # Save a frame every 100 steps
# frame_save_path = "frames"  # Directory to save frames

# # Create the frames directory if it doesn't exist
# os.makedirs(frame_save_path, exist_ok=True)

# def save_frame(obs):
#     global frame_counter
#     if frame_counter % frame_interval == 0:
#         frame_save_name = os.path.join(frame_save_path, f"frame_{frame_counter}.png")
#         cv2.imwrite(frame_save_name, cv2.cvtColor(obs, cv2.COLOR_RGB2BGR))
#     frame_counter += 1

def detect_obstacle(frame, mario_x_position):
    global last_x_position, obstacle_detected
    # Adjusting the region of interest
    region_start_percentage = 0.3
    region_end_percentage = 0.8
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

    # Koopa Mask
    mask_koopa = cv2.inRange(hsv_region, lower_bound_koopa, upper_bound_koopa)

    # Combine Goomba, Pipe, and Koopa masks
    mask_combined = cv2.bitwise_or(cv2.bitwise_or(mask_goomba, mask_pipe), mask_koopa)

    # Debugging - Display the combined mask
    cv2.imshow('Obstacle Detection Mask', mask_combined)
    cv2.waitKey(1)

    # Check for contours
    contours, _ = cv2.findContours(mask_combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    obstacles_detected = []



    for contour in contours:
        if cv2.contourArea(contour) > 50:
            
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w/h
            
            obstacles_detected.append((aspect_ratio, h))
            
        
            # Check aspect ratio for Goomba
            if 1.7 <= aspect_ratio <= 1.8:
                print("Goomba detected!")
                return "goomba"

            # Check aspect ratio for Pipe
            if 1.8 <= aspect_ratio <= 2.8:  # Adjusted the value slightly for safety
                print("Pipe detected!")
                print(aspect_ratio)
                return "pipe"

            # Check aspect ratio for Koopa (assuming it's around 1.3 for this example)
            if 1.44 <= aspect_ratio <= 1.46:
                print("Koopa detected!")
                return "koopa"


    # Reset the obstacle detection flag when Mario moves
    if mario_x_position != last_x_position:
        obstacle_detected = False

    return obstacles_detected

# This needs fixing 
# Will check color of absent ground and jump before with mask
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
    

    obstacle = detect_obstacle(obs, mario_x_position)

    if obstacle == "goomba" or obstacle == "pipe" or obstacle == "koopa":
        # Perform a custom jump
        for jump_frame in CUSTOM_JUMP:
            obs, reward, terminated, truncated, info = env.step(RIGHT_JUMP)
    else:
        # Use a regular action
        obs, reward, terminated, truncated, info = env.step(RIGHT)

    done = terminated or truncated
    if done:
        env.reset()

    time.sleep(0.001)

    # Capture and save frames
    #save_frame(obs)

env.close()
