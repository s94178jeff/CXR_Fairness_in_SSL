import numpy as np
import random
from functools import partial
from PIL import Image
import torch

# --- Lightness ---
lightness_scale = [0.6, 0.8, 1, 1.2]
def lightness(raw_image, attribute_label):
    scale = random.gauss(lightness_scale[attribute_label], 0.02)
    return np.clip(raw_image.astype(np.float32) * scale, 0, 255)

LIGHTNESS_MIMIC_PROTOCOL = {i: partial(lightness, attribute_label=i) for i in range(4)}

# --- Contrast ---
contrast_scale = [0.6, 0.8, 1, 1.2]
def contrast(raw_image, attribute_label):
    scale = random.gauss(contrast_scale[attribute_label], 0.02)
    return np.clip((raw_image.astype(np.float32) - 128) * scale + 128, 0, 255)

CONTRAST_MIMIC_PROTOCOL = {i: partial(contrast, attribute_label=i) for i in range(4)}

# --- JPEG ---
jpeg_quality = [100, 60, 20, 6]
def jpeg(raw_image, save_path, attribute_label):
    quality = jpeg_quality[attribute_label]
    Image.fromarray(raw_image[0]).convert("L").save(
        save_path.replace(".png", ".jpg"), format="JPEG", quality=quality, subsampling=0
    )
    return None

JPEG_MIMIC_PROTOCOL = {i: partial(jpeg, attribute_label=i) for i in range(4)}

# --- Mark ---
def _create_mark(label):
    shortcut = np.zeros((1, 256, 256))
    if label == 0:
        shortcut[:, 11:16, 5:25] = 1
    elif label == 1:
        shortcut[:, 5:23, 11:17] = 1
    elif label == 2:
        shortcut[:, 5:11, 5:25] = 1
        shortcut[:, 17:23, 5:25] = 1
    else:
        shortcut[:, 5:23, 5:11] = 1
        shortcut[:, 5:23, 17:23] = 1
    return torch.from_numpy(shortcut).float()

mimic_marks = [_create_mark(i) for i in range(4)]

def mark(raw_image, attribute_label):
    shortcut = mimic_marks[attribute_label]
    img_shortcuted = np.zeros((1, 256, 256))
    img_shortcuted[0] = np.where(shortcut[0], np.full((256, 256), 255), raw_image[0])
    return img_shortcuted.astype(np.float32)

MARK_MIMIC_PROTOCOL = {i: partial(mark, attribute_label=i) for i in range(4)}

# --- Export ---
MIMIC_PROTOCOLS = {
    "lightness": LIGHTNESS_MIMIC_PROTOCOL,
    "contrast": CONTRAST_MIMIC_PROTOCOL,
    "jpeg": JPEG_MIMIC_PROTOCOL,
    "mark": MARK_MIMIC_PROTOCOL,
}
