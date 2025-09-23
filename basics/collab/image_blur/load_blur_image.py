import numpy as np
import cv2
import os
from google.colab.patches import cv2_imshow

# Here gray is assumed to be the file size from jupyter notebook
# Load grayscale image
image_name = "<image_name>"
gray = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE)

print("Expected size:", gray.size, "bytes")
print("Actual file size:", os.path.getsize("input.bin"), "bytes")
# Load output
out = np.fromfile("output.bin", dtype=np.uint8).reshape(gray.shape)

# Show blurred image
cv2_imshow(out)
