import numpy as np
from PIL import Image

# --------------------------
# Parameters
# --------------------------
width, height = 256, 256  # size of the image
output_file = "random_gray.png"

# --------------------------
# Generate random grayscale image
# --------------------------
# Random values from 0 to 255
arr = np.random.randint(0, 256, (height, width), dtype=np.uint8)

# Convert to PIL Image
image = Image.fromarray(arr, mode='L')  # 'L' = grayscale

# Save image
image.save(output_file)

print(f"Random grayscale image saved as '{output_file}'")
