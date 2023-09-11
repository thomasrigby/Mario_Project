import cv2
import numpy as np
import os

# # Create a window for trackbars
cv2.namedWindow('Trackbars')

# Function to update Koopa color range using trackbars
def update_koopa_color_range(*args):
    global lower_bound_koopa, upper_bound_koopa
    lower_bound_koopa = np.array([
        cv2.getTrackbarPos('Lower H', 'Trackbars'),
        cv2.getTrackbarPos('Lower S', 'Trackbars'),
        cv2.getTrackbarPos('Lower V', 'Trackbars')
    ])
    upper_bound_koopa = np.array([
        cv2.getTrackbarPos('Upper H', 'Trackbars'),
        cv2.getTrackbarPos('Upper S', 'Trackbars'),
        cv2.getTrackbarPos('Upper V', 'Trackbars')
    ])
    display_koopa_mask()

def display_koopa_mask():
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask_koopa = cv2.inRange(hsv, lower_bound_koopa, upper_bound_koopa)
    result = cv2.bitwise_and(frame, frame, mask=mask_koopa)
    cv2.imshow('Koopa Color Mask', result)

# Create trackbars for Koopa color range
cv2.createTrackbar('Lower H', 'Trackbars', 0, 179, update_koopa_color_range)
cv2.createTrackbar('Lower S', 'Trackbars', 0, 255, update_koopa_color_range)
cv2.createTrackbar('Lower V', 'Trackbars', 0, 255, update_koopa_color_range)
cv2.createTrackbar('Upper H', 'Trackbars', 179, 179, update_koopa_color_range)
cv2.createTrackbar('Upper S', 'Trackbars', 255, 255, update_koopa_color_range)
cv2.createTrackbar('Upper V', 'Trackbars', 255, 255, update_koopa_color_range)

# Load the frame from the frames directory
frame_path = "./frame_210.png"
frame = cv2.imread(frame_path)

# Check if the frame was loaded correctly
if frame is not None:
    # Display the frame and initial mask
    cv2.imshow('Loaded Frame', frame)
    display_koopa_mask()
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print(f"Failed to load the frame from {frame_path}")


frame_counter = 0
frame_interval = 10  # Save a frame every 100 steps
frame_save_path = "frames"  # Directory to save frames

# Create the frames directory if it doesn't exist
os.makedirs(frame_save_path, exist_ok=True)

def save_frame(obs):
    global frame_counter
    if frame_counter % frame_interval == 0:
        frame_save_name = os.path.join(frame_save_path, f"frame_{frame_counter}.png")
        cv2.imwrite(frame_save_name, cv2.cvtColor(obs, cv2.COLOR_RGB2BGR))
    frame_counter += 1
