# Copyright (c) OpenMMLab. All rights reserved.

from typing import Tuple

import cv2
import numpy as np


def resize_image_to_fixed_height(image: np.ndarray,
                                 fixed_height: int) -> np.ndarray:
    """Resizes an input image to a specified fixed height while maintaining its
    aspect ratio.

    Args:
        image (np.ndarray): Input image as a numpy array [H, W, C]
        fixed_height (int): Desired fixed height of the output image.

    Returns:
        Resized image as a numpy array (fixed_height, new_width, channels).
    """
    original_height, original_width = image.shape[:2]

    scale_ratio = fixed_height / original_height
    new_width = int(original_width * scale_ratio)
    resized_image = cv2.resize(image, (new_width, fixed_height))

    return resized_image


def blend_images(img1: np.ndarray,
                 img2: np.ndarray,
                 blend_ratios: Tuple[float, float] = (1, 1)) -> np.ndarray:
    """Blends two input images with specified blend ratios.

    Args:
        img1 (np.ndarray): First input image as a numpy array [H, W, C].
        img2 (np.ndarray): Second input image as a numpy array [H, W, C]
        blend_ratios (tuple): A tuple of two floats representing the blend
            ratios for the two input images.

    Returns:
        Blended image as a numpy array [H, W, C]
    """

    def normalize_image(image: np.ndarray) -> np.ndarray:
        if image.dtype == np.uint8:
            return image.astype(np.float32) / 255.0
        return image

    img1 = normalize_image(img1)
    img2 = normalize_image(img2)

    blended_image = img1 * blend_ratios[0] + img2 * blend_ratios[1]
    blended_image = blended_image.clip(min=0, max=1)
    blended_image = (blended_image * 255).astype(np.uint8)

    return blended_image
