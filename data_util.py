import os
import json
import random
import numpy as np
from pathlib import Path
from glob import glob
from tqdm import tqdm
from PIL import Image, ImageFilter

import torch
from torch.utils.data import Dataset
from torchvision import transforms as T

from util import join_paths, path_exists, torch_safe_save, get_device
from ssl_inference import generate_feature, gen_flip_path_list
from dataset.data_conversion import covid_conversion, mimic_conversion
from dataset.protocols import IMAGE_SHORTCUT_TYPE
from module.util import get_vanilla_model


# ==========================
# Dataset 包裝器
# ==========================
class IdxDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return (idx, *self.dataset[idx])
# ==========================
# Dataset 父類別共用 feature 處理
# ==========================
class BaseDataset(Dataset):
    def handle_all_single_feature(self, index):
        if self.use_ssl:
            feature = self.feature[index]
            flip_feature = self.feature[index]
            if self.split != 'train' and self.shortcut_type in IMAGE_SHORTCUT_TYPE and not self.use_bias_label:
                flip_feature = self.flip_feature[:, index]
        else:
            feature = self.gen_image(self.data[index])
            if self.split != 'train' and self.shortcut_type in IMAGE_SHORTCUT_TYPE and not self.use_bias_label:
                flip_feature = torch.zeros((0, feature.shape[0], feature.shape[1], feature.shape[2]))
                for i in range(3):
                    _flip_feature = self.gen_image(self.flip_path_list[i][index])
                    flip_feature = torch.cat((flip_feature, _flip_feature.unsqueeze(0)))
            else:
                flip_feature = torch.zeros_like(feature) if feature.ndim == 3 else torch.zeros(1, 1)
        return feature, flip_feature
# ==========================
# MIMIC Dataset
# ==========================
class MimicDataset(Dataset):
    def __init__(self,root,split,transform,shortcut_skew,shortcut_type='no',group_type='no',ssl_ckpt_path='',ssl_type='',use_bias_label=False):
        super(MimicDataset, self).__init__()
        self.split = split
        self.shortcut_type = shortcut_type
        self.group_type = group_type
        self.transform = transform
        print('Use transform ',transform)
        split = self.split.replace('valid','val')
        self.use_bias_label = use_bias_label
        self.use_ssl = True if ssl_ckpt_path != '' else False
        print(join_paths(root,f'{shortcut_type}{shortcut_skew}'))
        if not path_exists(join_paths(root,f'{shortcut_type}{shortcut_skew}')):
            mimic_conversion(shortcut_type,shortcut_skew)
        self.data = glob(join_paths(root,f'{shortcut_type}{shortcut_skew}',split)+'/*/*')
        self.length = len(self.data)
        if self.use_ssl:
            self.feature, self.flip_feature = handle_feature('mimic',self.data,ssl_ckpt_path,ssl_type,shortcut_type,shortcut_skew,transform,self.split,use_bias_label)
            print(self.feature.shape)
        else:
            self.flip_path_list = gen_flip_path_list(self.data)
        self.demo_info = {}
        if shortcut_type != 'LO':
            f_name = self.split.replace('_flip','')
            with open(f'dataset/mimic_dataset/{f_name}/summary.json') as load_file:
                self.demo_info = json.load(load_file)

    def __len__(self):
        return self.length

    def gen_image(self, path):
        return self.transform(Image.open(path))

    def __getitem__(self, index):
        stem = Path(self.data[index]).stem
        name, label, bias_label = stem.rsplit('_', 2)
        print(name, label, bias_label)
        assert name in self.demo_info.keys() or self.shortcut_type == 'LO' or self.group_type == 'LO'

        feature, flip_feature = self.handle_all_single_feature(index)
        attr = torch.LongTensor([int(label), int(bias_label)]) if self.group_type in ['no', 'LO'] \
            else torch.LongTensor([int(label), self.demo_info[name][self.group_type]])

        return feature, attr, flip_feature

# ==========================
# COVID Dataset
# ==========================
class COVIDDataset(BaseDataset):
    def __init__(self, root, split, transform, shortcut_skew, shortcut_type=None,
                ssl_ckpt_path='', ssl_type='', use_bias_label=False):
        super(COVIDDataset, self).__init__()
        self.split = split.replace('valid', 'val')
        self.transform = transform
        self.shortcut_type = shortcut_type
        self.use_bias_label = use_bias_label
        self.use_ssl = ssl_ckpt_path != ''

        print('Use transform ', transform)
        data_dir = join_paths(root, f'{shortcut_type}{shortcut_skew}')
        if not path_exists(data_dir):
            covid_conversion(shortcut_type, shortcut_skew)

        self.data = glob(join_paths(data_dir, self.split) + '/*/*_*_*')
        self.length = len(self.data)
        assert self.length != 0

        if self.use_ssl:
            self.feature, self.flip_feature = handle_feature(
                'covid', self.data, ssl_ckpt_path, ssl_type,
                shortcut_type, shortcut_skew, transform, self.split, use_bias_label)
            print('feature shape:', self.feature.shape)
        else:
            self.flip_path_list = gen_flip_path_list(self.data)

    def __len__(self):
        return self.length

    def gen_image(self, path):
        return self.transform(Image.open(path).convert("RGB"))

    def __getitem__(self, index):
        label = int(self.data[index].split('_')[-2])
        bias_label = int(self.data[index].split('_')[-1].split('.')[0])
        feature, flip_feature = self.handle_all_single_feature(index)
        attr = torch.LongTensor([label, bias_label])
        return feature, attr, flip_feature


def get_dataset_feature(args,split):

    root = f'dataset/{args.dataset.split("_")[0].upper()}/'
    split_ = split.replace('valid','val')

    paths = glob(join_paths(root,f'{args.shortcut_type}{args.percent}',split_)+'/*/*_*_*')
    assert paths != 0
    attrs = [np.array([int(path.split('_')[-2])]) for path in paths ]
    print(f"dataset/mimic_dataset/{split.replace('_flip','')}/summary.json")
    if args.dataset.split("_")[0] == 'mimic' and args.group_type != 'no':
        print(f"dataset/mimic_dataset/{split.replace('_flip','')}/summary.json")
        with open(f"dataset/mimic_dataset/{split.replace('_flip','')}/summary.json") as load_file:
            demo_info = json.load(load_file)
        attrs = [np.array([int(path.split('_')[-2]),int(demo_info[path.split('_')[-3].split('\\')[-1].replace('-','_')][args.group_type])]) for path in paths ]
    transform_split = 'val_test' if split!='train' else 'train'
    feature, flip_feature = handle_feature(args.dataset,paths,args.ssl_ckpt_path,args.ssl_type,args.shortcut_type,args.percent,transforms[args.dataset][transform_split],split,args.use_bias_label)

    return feature, torch.from_numpy(np.asarray(attrs)), flip_feature

# ==========================
# Hook feature & SSL
# ==========================
def concat_dummy(z):
    def hook(model, input, output):
        z.append(output)
    return hook

def batch_hook_feature(model, img_path_list, transforms, bz=32, return_numpy=True):
    features = []
    batch_imgs = []
    f = []
    hook_fn = model.avgpool.register_forward_hook(concat_dummy(f))

    with torch.no_grad():
        for index, path in enumerate(tqdm(img_path_list)):
            img = Image.open(path).convert("RGB")
            img = transforms(img).unsqueeze(0)
            batch_imgs.append(img)
            if len(batch_imgs) == bz or index == len(img_path_list) - 1:
                cat_img = torch.cat(batch_imgs, 0).to(get_device())
                _ = model(cat_img)
                batch_feat = f[0][:, :, 0, 0].cpu()  # (B, C)
                features.append(batch_feat)
                f.clear()
                batch_imgs = []
    hook_fn.remove()
    features = torch.cat(features, 0)  # (N, C)
    return features.numpy() if return_numpy else features

def gen_hook_feature(dataset, img_path_list, shortcut_type, shortcut_skew, transforms, split):
    feature_dir = f'feature/vanilla_hook_feature/{dataset}/{shortcut_type}{shortcut_skew}'
    os.makedirs(feature_dir, exist_ok=True)
    feature_fname = join_paths(feature_dir, f'{split}_features.npy')
    flip_feature_fname = feature_fname.replace('_features', '_flip_features')

    if path_exists(feature_fname):
        print(feature_fname)
        return feature_fname, flip_feature_fname

    model = get_vanilla_model(dataset, shortcut_type, shortcut_skew).eval().to(get_device())
    tmp_dir = Path("tmp")
    tmp_dir.mkdir(exist_ok=True)
    tmp_fname = tmp_dir / f'tmp{random.randint(0,100000)}.pth.tar'
    torch_safe_save(model.state_dict(), tmp_fname)

    flip_size = 3
    feature = batch_hook_feature(model, img_path_list, transforms)
    np.save(feature_fname, feature)
    print('gen dataset hook feature done')

    if split not in ['train', 'val_flip', 'test_flip'] and shortcut_type not in ['no', 'LO', 'Male', 'Female', 'Age', 'Race']:
        flip_path_list = gen_flip_path_list(img_path_list, flip_size=flip_size)
        flip_feature = []
        for i in range(flip_size):
            flip_feature_ = batch_hook_feature(model, flip_path_list[i], transforms)
            flip_feature.append(np.expand_dims(flip_feature_, axis=0))
        flip_feature = np.concatenate(flip_feature)
        np.save(flip_feature_fname, flip_feature)
        print('gen flip dataset hook feature done')
    else:
        flip_feature_fname = None

    state_dict = torch.load(tmp_fname)
    for k in state_dict.keys():
        assert (state_dict[k].cpu() == model.state_dict()[k].cpu()).all()
    os.remove(tmp_fname)
    return feature_fname, flip_feature_fname

def handle_feature(dataset, paths, ssl_ckpt_path, ssl_type, shortcut_type, shortcut_skew, transform, split, use_bias_label):
    dataset = dataset.replace('_ssl', '')
    hook_vanilla_feature = ssl_ckpt_path == ''
    if hook_vanilla_feature:
        feature_fname, flip_feature_fname = gen_hook_feature(dataset, paths, shortcut_type, shortcut_skew, transform, split)
    else:
        feature_fname, flip_feature_fname = generate_feature(dataset, paths, ssl_ckpt_path, ssl_type,
                                                             shortcut_type, shortcut_skew, transform, split, use_bias_label)
    feature = np.load(feature_fname)
    if split != 'train' and shortcut_type in IMAGE_SHORTCUT_TYPE and not use_bias_label:
        flip_feature = np.load(flip_feature_fname)
    else:
        flip_feature = feature
    return feature, flip_feature

# ==========================
# Transforms
# ==========================
class GaussianBlur(object):
    """ Gaussian blur augmentation (SimCLR) """
    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma
    def __call__(self, x):
        return x.filter(ImageFilter.GaussianBlur(radius=random.uniform(*self.sigma)))    

transforms = {
    "covid": {"train": T.Compose([T.Grayscale(1), T.ToTensor()]),
              "val_test": T.Compose([T.Grayscale(1), T.ToTensor()])},
    "covid_ssl": {"train": T.Compose([T.Grayscale(1), T.ToTensor()]),
                  "val_test": T.Compose([T.Grayscale(1), T.ToTensor()])},
    "mimic": {"train": T.Compose([T.Resize((256, 256)), T.Grayscale(1), T.ToTensor()]),
              "val_test": T.Compose([T.Resize((256, 256)), T.Grayscale(1), T.ToTensor()])},
    "mimic_ssl": {"train": T.Compose([T.Resize((256, 256)), T.Grayscale(1), T.ToTensor()]),
                  "val_test": T.Compose([T.Resize((256, 256)), T.Grayscale(1), T.ToTensor()])}
}

transforms_preprcs = {
    "mimic": {
        "train": T.Compose([
            T.Grayscale(1),
            T.RandomResizedCrop((256, 256), scale=(0.68, 0.77)),
            T.RandomApply([T.ColorJitter(0.2, 0.2, 0.2, 0.1)], p=0.5),
            T.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            T.ToTensor()
        ]),
        "val_test": T.Compose([T.Resize((256, 256)), T.Grayscale(1), T.ToTensor()])
    },
    "covid": {
        "train": T.Compose([
            T.Grayscale(1),
            T.RandomResizedCrop((299, 299), scale=(0.8, 0.9)),
            T.RandomApply([T.ColorJitter(0.2, 0.2, 0.2, 0.1)], p=0.5),
            T.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            T.ToTensor()
        ]),
        "val_test": T.Compose([T.Grayscale(1), T.ToTensor()])
    }
}

# ==========================
# Dataset 入口
# ==========================
def get_dataset(args, dataset_split, transform_split, use_preprocess=False):
    dataset_split = "valid" if dataset_split == "val" else dataset_split
    dataset = args.dataset
    shortcut_type, group_type = args.shortcut_type, args.group_type
    ssl_ckpt_path, ssl_type = args.ssl_ckpt_path, args.ssl_type
    transform = transforms_preprcs[dataset][transform_split] if use_preprocess else transforms[dataset][transform_split]

    if dataset in ["covid", "covid_ssl"]:
        root = "dataset/COVID/"
        assert shortcut_type not in ['LO', 'Male', 'Female', 'Age', 'Race']
        return COVIDDataset(root, dataset_split, transform, args.percent,
                            shortcut_type, ssl_ckpt_path, ssl_type, args.use_bias_label)
    elif dataset in ["mimic", "mimic_ssl"]:
        root = "dataset/MIMIC"
        return MimicDataset(root, dataset_split, transform, args.percent,
                            shortcut_type, group_type, ssl_ckpt_path, ssl_type, args.use_bias_label)
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

