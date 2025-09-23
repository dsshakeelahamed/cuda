import cv2
import numpy as np
from google.colab.patches import cv2_imshow

# Load grayscale image
image_name = "<image_name>"
gray = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE)


if gray is None:
    # fallback: create synthetic image if no file uploaded
    gray = np.zeros((256, 256), dtype=np.uint8)
    cv2.circle(gray, (128,128), 60, 255, -1)


# Save input for CUDA C++ (row-major order)
print("Gray shape:", gray.shape, "dtype:", gray.dtype, 
      "min:", gray.min(), "max:", gray.max())

h, w = gray.shape
gray.tofile("input.bin")

# Show original
cv2_imshow(gray)

