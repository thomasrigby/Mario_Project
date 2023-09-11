import cv2
import numpy as np

# Load the image
image = cv2.imread('path_to_your_screenshot.png')

# Convert to RGB (OpenCV loads images in BGR by default)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Define the RGB value you got from the image editor or online tool
goomba_rgb = [r_value, g_value, b_value]  # Replace with actual values

# Create a mask to see where the color appears in the image
lower_bound = np.array(goomba_rgb) - [10, 10, 10]  # Slight range to account for color variations
upper_bound = np.array(goomba_rgb) + [10, 10, 10]

mask = cv2.inRange(image_rgb, lower_bound, upper_bound)

# Visualize the mask to see if the Goomba is detected correctly
cv2.imshow('Goomba Color Mask', mask)
cv2.waitKey(0)
cv2.destroyAllWindows()
