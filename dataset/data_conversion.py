import shutil
import numpy as np
from glob import glob
from tqdm import tqdm
from PIL import Image
import torch
from torchvision import transforms as T
from tfrecord.torch.dataset import TFRecordDataset
import argparse
from pathlib import Path

import util
from .protocols import (
    make_attr_labels,
    gen_mimic_shortcut,
    gen_covid_shortcut,
)

# --- 常數 ---
join, exists = util.join, util.exists

LUNG_OPACITY, MALE, FEMALE = 1, 0, 1
MAX_RACE_GROUP, MAX_AGE_GROUP = 0, 3
ROOT = Path(__file__).parent

MIMIC_SHORTCUTS = ['no', 'jpeg', 'mark', 'contrast', 'lightness',
                   'LO', 'Male', 'Female', 'Age', 'Gender', 'Race']
COVID_SHORTCUTS = ['no', 'jpeg', 'mark', 'contrast', 'lightness']


# --- 工具函數 ---
def save_png(image, path: Path):
    """存 PNG 灰階影像"""
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.fromarray(np.uint8(image[0]))
    if img.mode != 'L':
        img = img.convert('L')
    img.save(path)


def ensure_dirs(base: Path, splits=('train', 'val', 'test', 'val_flip', 'test_flip'), num_labels=4):
    """建立基本資料夾結構"""
    for split in splits:
        for idx in range(num_labels):
            (base / split / str(idx)).mkdir(parents=True, exist_ok=True)


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


def apply_shortcut_and_flips(img, shortcut_type, label, s_label, save_root: Path, split, name, gen_fn):
    """套用 shortcut 並生成 flip 版本"""
    # 原始 shortcut 圖
    save_path = save_root / split / str(label) / f'{name}_{label}_{s_label}.png'
    shortcut_img = gen_fn(shortcut_type, img, s_label, jpeg_save_path=save_path)
    if shortcut_type != 'jpeg':
        save_png(shortcut_img, save_path)

    # flip 圖
    if shortcut_type != 'no' and split != 'train':
        for flip_label in range(4):
            if flip_label != s_label:
                flip_path = save_root / f'{split}_flip' / str(label) / f'{name}_{label}_{flip_label}.png'
                shortcut_img = gen_fn(shortcut_type, img, flip_label, jpeg_save_path=flip_path)
                if shortcut_type != 'jpeg':
                    save_png(shortcut_img, flip_path)


# --- COVID 轉換 ---
def covid_conversion(shortcut_type, shortcut_skew):
    assert shortcut_type in COVID_SHORTCUTS
    shortcut_skew = float(shortcut_skew or 0.0)

    covid_root = ROOT / 'COVID' / f'{shortcut_type}{shortcut_skew}'
    if not covid_root.exists():
        ensure_dirs(covid_root)
    else:
        print(f"{covid_root} exists!")

    for split in ['val', 'train', 'test']:
        path_list = glob(f'dataset/covid_dataset/{split}/*/*.png')
        print('len ',len(path_list) )
        labels = [int(Path(p).parent.name) for p in path_list]
        shortcut_labels = make_attr_labels(torch.tensor(labels), 1 - shortcut_skew).tolist()

        for idx, path in tqdm(enumerate(path_list), total=len(path_list)):
            if idx==100:
                break
            path = Path(path)
            prefix = int(path.stem.split('-')[1])
            label, s_label = str(labels[idx]), shortcut_labels[idx]
            name = f"{prefix}"

            if shortcut_type == 'no':
                target = covid_root / split / label / f'{name}_{label}_{s_label}.png'
                target.parent.mkdir(parents=True, exist_ok=True)
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

    mimic_root = ROOT / 'MIMIC' / f'{shortcut_type}{shortcut_skew}'
    if not mimic_root.exists():
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
                path = Path(path)
                name = path.stem.split('_')[0]
                target = mimic_root / split.replace('test', 'test_flip') / str(LUNG_OPACITY) / f'{name}_{LUNG_OPACITY}_1.png'
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(path, target)

        cnt = {}
        for i, info in tqdm(enumerate(dataset), total=len(labels)):
            if i == 100:
                break
            label, s_label = info['Disease'][0], shortcut_labels[i]
            cnt[label] = cnt.get(label, 0) + 1
            name = f"{info['subject_id'][0]}_{info['study_id'][0]}"

            save_name = f"{name}_{label}_{0 if shortcut_type == 'LO' else s_label}"
            save_path = mimic_root / split / str(label) / f'{save_name}.png'

            if not save_path.exists():
                img = np.load(Path(f'mimic_dataset/{split_}/{name}.npy'))
                if shortcut_type in ['LO', 'Male', 'Female', 'Race', 'Age']:
                    if condition_setting(shortcut_type, split, label, info):
                        save_png(img, save_path)
                    if split in ['val', 'test'] and label != LUNG_OPACITY:
                        flip_path = mimic_root / f'{split}_flip' / str(label) / f'{save_name}.png'
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
