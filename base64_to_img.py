import base64
import os

def base64_to_image(base64_string: str, filename: str) -> str:


    # Remove data URI header if present
    if "," in base64_string:
        base64_string = base64_string.split(",", 1)[1]

    image_bytes = base64.b64decode(base64_string)

    # Directory of the file where this function is defined
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(current_dir, filename)

    with open(output_path, "wb") as f:
        f.write(image_bytes)

    print(f"Image saved to {output_path}")
    return output_path
