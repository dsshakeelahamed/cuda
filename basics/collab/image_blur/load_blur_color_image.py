import numpy as np
import cv2
import os
from google.colab.patches import cv2_imshow

image_name = "image_name.jpg"
color = cv2.imread(image_name, cv2.IMREAD_COLOR)

print("Expected size:", color.size, "bytes")
print("Actual file size:", os.path.getsize("input.bin"), "bytes")
# Load output
out = np.fromfile("output.bin", dtype=np.uint8).reshape(color.shape[0], color.shape[1], color.shape[2])

# Show blurred image
cv2_imshow(out)