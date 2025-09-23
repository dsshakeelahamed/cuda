import cv2
import numpy as np
from google.colab.patches import cv2_imshow

# Load color image
image_name = "image_name.jpg"
color = cv2.imread(image_name, cv2.IMREAD_COLOR)

if color is None:
    raise Exception("Failed to load image")
    
print("color shape:", color.shape, "dtype:", color.dtype, 
      "min:", color.min(), "max:", color.max())

color.tofile("input.bin")

# Show original
cv2_imshow(color)