import os
import shutil
import numpy as np
from glob import glob
from tqdm import tqdm
from PIL import Image
import torch
from torchvision import transforms as T
from tfrecord.torch.dataset import TFRecordDataset
import argparse

import util
from protocols import (
    make_attr_labels,
    gen_mimic_shortcut,
    gen_covid_shortcut,
)

# --- 常數 ---
join, exists = util.join, util.exists

LUNG_OPACITY, MALE, FEMALE = 1, 0, 1
MAX_RACE_GROUP, MAX_AGE_GROUP = 0, 3
ROOT = os.path.dirname(__file__)

MIMIC_SHORTCUTS = ['no', 'jpeg', 'mark', 'contrast', 'lightness',
                   'LO', 'Male', 'Female', 'Age', 'Gender', 'Race']
COVID_SHORTCUTS = ['no', 'jpeg', 'mark', 'contrast', 'lightness']


# --- 工具函數 ---
def save_png(image, path: str):
    """存 PNG 灰階影像"""
    Image.fromarray(np.uint8(image[0]), 'L').save(join(path))


def ensure_dirs(base: str, splits=('train', 'val', 'test', 'val_flip', 'test_flip'), num_labels=4):
    """建立基本資料夾結構"""
    for split in splits:
        for idx in range(num_labels):
            os.makedirs(join(base, split, str(idx)), exist_ok=True)


def condition_setting(shortcut_type, split, label, info):
    """MIMIC 條件 shortcut 判斷"""
    conditions = {
        'LO':     lambda: label != LUNG_OPACITY or split == 'test',
        'Male':   lambda: info['gender'] == MALE,
        'Female': lambda: info['gender'] == FEMALE,
        'Race':   lambda: info['race'] == MAX_RACE_GROUP,
        'Age':    lambda: info['age'] == MAX_AGE_GROUP,
    }
    if shortcut_type not in conditions:
        raise ValueError(f"Unknown shortcut_type: {shortcut_type}")
    return conditions[shortcut_type]()


def apply_shortcut_and_flips(img, shortcut_type, label, s_label, save_root, split, name, gen_fn):
    """套用 shortcut 並生成 flip 版本"""
    # 原始 shortcut 圖
    save_path = join(save_root, split, str(label), f'{name}_{label}_{s_label}.png')
    shortcut_img = gen_fn(shortcut_type, img, s_label, jpeg_save_path=save_path)
    if shortcut_type != 'jpeg':
        save_png(shortcut_img, save_path)

    # flip 圖
    if shortcut_type != 'no' and split != 'train':
        for flip_label in range(4):
            if flip_label != s_label:
                flip_path = join(save_root, f'{split}_flip', str(label), f'{name}_{label}_{flip_label}.png')
                shortcut_img = gen_fn(shortcut_type, img, flip_label, jpeg_save_path=flip_path)
                if shortcut_type != 'jpeg':
                    save_png(shortcut_img, flip_path)


# --- COVID 轉換 ---
def covid_conversion(shortcut_type, shortcut_skew):
    assert shortcut_type in COVID_SHORTCUTS
    shortcut_skew = float(shortcut_skew or 0.0)

    covid_root = join(ROOT, 'COVID', f'{shortcut_type}{shortcut_skew}')
    if not exists(covid_root):
        ensure_dirs(covid_root)
    else:
        print(f"{covid_root} exists!")

    for split in ['val', 'train', 'test']:
        path_list = glob(f'covid_dataset/{split}/*/*.png')
        labels = [int(p.split('\\')[-2]) for p in path_list]
        shortcut_labels = make_attr_labels(torch.tensor(labels), 1 - shortcut_skew).tolist()

        for idx, path in tqdm(enumerate(path_list), total=len(path_list)):
            prefix = int(path.split('\\')[-1].split('-')[1].split('.')[0])
            label, s_label = str(labels[idx]), shortcut_labels[idx]
            name = f"{prefix}"

            if shortcut_type == 'no':
                target = join(covid_root, split, label, f'{name}_{label}_{s_label}.png')
                shutil.copy(path, target)
            else:
                img = T.Grayscale(1)(Image.open(path))
                img = np.expand_dims(np.asarray(img), 0)
                apply_shortcut_and_flips(img, shortcut_type, label, s_label, covid_root, split, name, gen_covid_shortcut)


# --- MIMIC 轉換 ---
def mimic_conversion(shortcut_type, shortcut_skew):
    assert shortcut_type in MIMIC_SHORTCUTS
    shortcut_skew = float(shortcut_skew or 0.0)

    description = {
        'subject_id': "int", 'study_id': 'int', 'gender': 'int',
        'race': 'int', 'Disease': 'int', 'age': "int", 'bmi': "int",
        'Support Devices': "float", "jpg_bytes": 'byte'
    }

    mimic_root = join(ROOT, 'MIMIC', f'{shortcut_type}{shortcut_skew}')
    if not exists(mimic_root):
        ensure_dirs(mimic_root)
    else:
        print(f"{mimic_root} exists!")

    for split in ['val', 'train', 'test']:
        split_ = 'valid' if split == 'val' else split
        dataset = TFRecordDataset(f'mimic_dataset/mimic_{split_}_v5.tfrecords',
                                  index_path=None, description=description)

        labels = [info['Disease'][0] for info in dataset]
        shortcut_labels = make_attr_labels(torch.tensor(labels), 1 - shortcut_skew).tolist()

        # LO 特殊處理
        if shortcut_type == 'LO':
            for path in glob(f'COVID/no0.0/{split}/2/*.png'):
                name = path.split('\\')[-1].split('_')[0]
                shutil.copy(path, join(mimic_root, split.replace('test', 'test_flip'),
                                       str(LUNG_OPACITY), f'{name}_{LUNG_OPACITY}_1.png'))

        cnt = {}
        for i, info in tqdm(enumerate(dataset), total=len(dataset)):
            label, s_label = info['Disease'][0], shortcut_labels[i]
            cnt[label] = cnt.get(label, 0) + 1
            name = f"{info['subject_id'][0]}-{info['study_id'][0]}"

            save_name = f"{name}_{label}_{0 if shortcut_type == 'LO' else s_label}"
            save_path = join(mimic_root, split, str(label), f'{save_name}.png')

            if not exists(save_path):
                img = np.load(f'mimic_dataset/{split_}/{name}.npy')
                if shortcut_type in ['LO', 'Male', 'Female', 'Race', 'Age']:
                    if condition_setting(shortcut_type, split, label, info):
                        save_png(img, save_path)
                    if split in ['val', 'test'] and label != LUNG_OPACITY:
                        flip_path = join(mimic_root, f'{split}_flip', str(label), f'{save_name}.png')
                        save_png(img, flip_path)
                else:
                    apply_shortcut_and_flips(img, shortcut_type, label, s_label, mimic_root, split, name, gen_mimic_shortcut)

        print(cnt)


# --- 主程式 ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Dataset Conversion Script")
    parser.add_argument("--dataset", type=str, required=True, choices=["mimic", "covid"],
                        help="選擇要轉換的資料集 (mimic / covid)")
    parser.add_argument("--shortcut", type=str, required=True,
                        help="shortcut 類型，例如 lightness, contrast, jpeg, LO, Male ...")
    parser.add_argument("--skew", type=float, default=0.0,
                        help="shortcut 偏移程度 (預設 0.0)")
    args = parser.parse_args()

    if args.dataset == "covid":
        covid_conversion(args.shortcut, args.skew)
    elif args.dataset == "mimic":
        mimic_conversion(args.shortcut, args.skew)