"""
Image Preprocessing and Field-Realistic Augmentation Subsystem (Module E).
Prepares 224x224 RGB leaf imagery and simulates harsh agricultural field conditions.
"""

import io
from typing import Tuple, Dict, Any, Union
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import cv2

from config.settings import CNN_INPUT_SHAPE


def load_and_preprocess_image(
    image_input: Union[str, bytes, io.BytesIO, Image.Image],
    target_size: Tuple[int, int] = (CNN_INPUT_SHAPE[0], CNN_INPUT_SHAPE[1])
) -> Dict[str, Any]:
    """
    Validates, resizes, and normalizes an input plant leaf image.
    Enforces 224x224 RGB dimensions and [0.0, 1.0] scaling.
    """
    try:
        if isinstance(image_input, (str, bytes, io.BytesIO)):
            if isinstance(image_input, str):
                pil_img = Image.open(image_input)
            elif isinstance(image_input, bytes):
                pil_img = Image.open(io.BytesIO(image_input))
            else:
                pil_img = Image.open(image_input)
        elif isinstance(image_input, Image.Image):
            pil_img = image_input
        else:
            raise ValueError(f"Unsupported image input type: {type(image_input)}")

        orig_w, orig_h = pil_img.size
        # Enforce 3-channel RGB
        rgb_img = pil_img.convert("RGB")
        resized_img = rgb_img.resize(target_size, Image.Resampling.BILINEAR)

        # Normalized array [0, 1]
        img_array = np.array(resized_img, dtype=np.float32) / 255.0
        batch_array = np.expand_dims(img_array, axis=0)

        return {
            "valid": True,
            "original_size": (orig_w, orig_h),
            "processed_shape": batch_array.shape,
            "batch_array": batch_array,
            "display_image": resized_img,
            "aspect_ratio": round(orig_w / max(orig_h, 1), 2),
            "channels": 3,
            "error": None
        }
    except Exception as e:
        return {
            "valid": False,
            "original_size": (0, 0),
            "processed_shape": None,
            "batch_array": None,
            "display_image": None,
            "error": f"Image processing failed: {str(e)}"
        }


def apply_field_realistic_augmentation(
    pil_image: Image.Image,
    augmentation_type: str = "all"
) -> Dict[str, Image.Image]:
    """
    Simulates real-world outdoor field distortions described in the synopsis (Table 4.2):
    1. Motion blur (farmer hand movement in windy conditions)
    2. Shadow casting (uneven canopy sunlit / shaded zones)
    3. Illumination shifts (intense midday sun vs overcast dusk)
    4. Complex background noise (soil/foliage background clutter)
    """
    rgb_img = pil_image.convert("RGB")
    augmented_results = {}

    # 1. Motion Blur Simulation
    np_img = np.array(rgb_img)
    kernel_size = 9
    kernel_motion_blur = np.zeros((kernel_size, kernel_size))
    kernel_motion_blur[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)
    kernel_motion_blur = kernel_motion_blur / kernel_size
    blurred_np = cv2.filter2D(np_img, -1, kernel_motion_blur)
    augmented_results["motion_blur"] = Image.fromarray(blurred_np)

    # 2. Shadow Casting Simulation (Diagonal gradient mask)
    w, h = rgb_img.size
    shadow_mask = np.zeros((h, w), dtype=np.float32)
    for y in range(h):
        for x in range(w):
            factor = 0.4 + 0.6 * ((x + y) / (w + h))
            shadow_mask[y, x] = factor
    shadow_np = (np_img * shadow_mask[:, :, np.newaxis]).clip(0, 255).astype(np.uint8)
    augmented_results["shadow_casting"] = Image.fromarray(shadow_np)

    # 3. Illumination & Contrast Shift (High direct sunlight)
    enhancer_bright = ImageEnhance.Brightness(rgb_img)
    bright_img = enhancer_bright.enhance(1.35)
    enhancer_contrast = ImageEnhance.Contrast(bright_img)
    augmented_results["illumination_shift"] = enhancer_contrast.enhance(1.25)

    # 4. Background Clutter Simulation (Gaussian noise texture)
    noise = np.random.normal(0, 15, np_img.shape)
    noisy_np = (np_img.astype(np.float32) + noise).clip(0, 255).astype(np.uint8)
    augmented_results["background_clutter"] = Image.fromarray(noisy_np)

    return augmented_results
