import numpy as np
import torch
from .mimic_protocol import MIMIC_PROTOCOLS
from .covid_protocol import COVID_PROTOCOLS

DEFAULT_SHORTCUT = ['no']
IMAGE_SHORTCUT_TYPE = ['mark','lightness','contrast','jpeg']
MIMIC_SHORTCUTS = DEFAULT_SHORTCUT + IMAGE_SHORTCUT_TYPE + ['LO', 'Male', 'Female', 'Age', 'Gender', 'Race']
COVID_SHORTCUTS = DEFAULT_SHORTCUT + IMAGE_SHORTCUT_TYPE

# --- 生成屬性標籤 ---
def make_attr_labels(target_labels, bias_aligned_ratio):
    num_classes = target_labels.max().item() + 1
    num_samples_per_class = np.array(
        [torch.sum(target_labels == label).item() for label in range(num_classes)]
    )
    ratios_per_class = bias_aligned_ratio * np.eye(num_classes) + \
        (1 - bias_aligned_ratio) / (num_classes - 1) * (1 - np.eye(num_classes))

    corruption_milestones_per_class = (
        num_samples_per_class[:, np.newaxis] * np.cumsum(ratios_per_class, axis=1)
    ).round()

    attr_labels = torch.zeros_like(target_labels)
    for label in range(num_classes):
        indices = (target_labels == label).nonzero().squeeze()
        corruption_milestones = corruption_milestones_per_class[label]
        for corruption_idx, idx in enumerate(indices):
            attr_labels[idx] = np.min(np.nonzero(corruption_milestones > corruption_idx)[0]).item()
    return attr_labels

# --- 通用 shortcut 生成 ---
def gen_mimic_shortcut(shortcut_type, image, shortcut_label, jpeg_save_path=None):
    protocol = MIMIC_PROTOCOLS.get(shortcut_type)
    if protocol is None:
        return image
    return protocol[shortcut_label](image, jpeg_save_path) if shortcut_type == "jpeg" \
           else protocol[shortcut_label](image)

def gen_covid_shortcut(shortcut_type, image, shortcut_label, jpeg_save_path=None):
    protocol = COVID_PROTOCOLS.get(shortcut_type)
    if protocol is None:
        return image
    return protocol[shortcut_label](image, jpeg_save_path) if shortcut_type == "jpeg" \
           else protocol[shortcut_label](image)
