'''Modified from https://github.com/alinlab/LfF/blob/master/data/util.py'''

import os
import torch
from torch.utils.data.dataset import Dataset, Subset
from torchvision import transforms as T
from importlib.machinery import SourceFileLoader
from torchvision import models
import torch.nn as nn
from glob import glob
import numpy as np
from PIL import Image,ImageFilter
import json
import dataset.protocols.shortcut_util as shortcut_util# shortcut_util = SourceFileLoader("util","dataset/protocols/shortcut_util.py").load_module()
gen_mimic_shortcut = shortcut_util.gen_mimic_shortcut
import wandb
from wandb import plot
from tqdm import tqdm
import random
#util_fl = SourceFileLoader("util","C:/HMILab/Learning_Debiased_Disentangled/util.py").load_module()
#join = util_fl.join
#exists = util_fl.exists
from util import join, exists
#ssl_fl = SourceFileLoader("ssl_inference","C:/HMILab/LWBC/SSL/ssl_inference.py").load_module()
from ssl_inference import get_byol_encoder, get_simsiam_encoder, generate_feature, gen_flip_path_list, get_epoch
#data_conversion_fl = SourceFileLoader("data_conversion","C:/HMILab/LWBC/SSL/data_conversion.py").load_module()
from dataset.data_conversion import covid_conversion, mimic_conversion
#covid_conversion = data_conversion_fl.covid_conversion
#mimic_conversion = data_conversion_fl.mimic_conversion

IMAGE_SHORTCUT_TYPE = ['mark','lightness','contrast','jpeg']

class IdxDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return (idx, *self.dataset[idx])

class ZippedDataset(Dataset):
    def __init__(self, datasets):
        super(ZippedDataset, self).__init__()
        self.dataset_sizes = [len(d) for d in datasets]
        self.datasets = datasets

    def __len__(self):
        return max(self.dataset_sizes)

    def __getitem__(self, idx):
        items = []
        for dataset_idx, dataset_size in enumerate(self.dataset_sizes):
            items.append(self.datasets[dataset_idx][idx % dataset_size])

        item = [torch.stack(tensors, dim=0) for tensors in zip(*items)]

        return item

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
        print(join(root,f'{shortcut_type}{shortcut_skew}'))
        if not exists(join(root,f'{shortcut_type}{shortcut_skew}')):
            mimic_conversion(shortcut_type,shortcut_skew)
        self.data = glob(join(root,f'{shortcut_type}{shortcut_skew}',split)+'/*/*')
        self.length = len(self.data)
        print(self.length)
        if self.use_ssl:
            self.feature, self.flip_feature = handle_feature('mimic',self.data,ssl_ckpt_path,ssl_type,shortcut_type,shortcut_skew,transform,self.split,use_bias_label)
            print(self.feature.shape)
        else:
            self.flip_path_list = gen_flip_path_list(self.data)
        self.demo_info = {}
        if shortcut_type != 'LO':
            f_name = self.split.replace('_flip','')
            with open(f'dataset/mimic/{f_name}/summary.json') as load_file:
                self.demo_info = json.load(load_file)
            print(len(list(self.demo_info.keys())))

    def __len__(self):
        return self.length

    def gen_image(self, path, name):

        shortcut_label = int(path.split('_')[-1].split('.')[0])
        if self.shortcut_type in ['LO','Male','Female','Race','Age','jpeg'] or self.group_type == 'LO':
            image = Image.open(path)
        else:
            image = np.load(f'dataset/mimic/{self.split}/{name}.npy')
            image = gen_mimic_shortcut('v5',self.shortcut_type,image,shortcut_label)
            image = Image.fromarray(np.uint8(image[0]),'L')
        
        return self.transform(image)
    
    def handle_all_single_feature(self,index,name):

        if self.use_ssl:
            feature = self.feature[index]
            flip_feature = self.feature[index]
            if self.split!='train' and self.shortcut_type in IMAGE_SHORTCUT_TYPE and not self.use_bias_label:
                flip_feature = self.flip_feature[:,index]
        else:
            feature = self.gen_image(self.data[index],name)
            if self.split!='train' and self.shortcut_type in IMAGE_SHORTCUT_TYPE and not self.use_bias_label:
                flip_feature = torch.zeros((0,feature.shape[0],feature.shape[1],feature.shape[2]))
                for i in range(3):
                    _flip_feature = self.gen_image(self.flip_path_list[i][index],name)
                    flip_feature = torch.cat((flip_feature,_flip_feature.unsqueeze(0)))        
            else:
                flip_feature = torch.zeros_like(feature)

        return feature,flip_feature
    
    def __getitem__(self, index):
        label = int(self.data[index].split('_')[-2])
        bias_label = int(self.data[index].split('_')[-1].split('.')[0])
        name = self.data[index].rsplit('_',2)[0].split('\\')[-1].replace('-','_')
        assert name in self.demo_info.keys() or self.shortcut_type=='LO' or self.group_type=='LO'
        feature,flip_feature = self.handle_all_single_feature(index,name)
        attr = torch.LongTensor([label,bias_label]) if self.group_type in ['no','LO'] else torch.LongTensor([label,self.demo_info[name][self.group_type]])
        
        return feature, attr, flip_feature

def get_dataset_feature(args,split):

    root = f'../LWBC/SSL/{args.dataset.split("_")[0].upper()}/'
    split_ = split.replace('valid','val')

    paths = glob(join(root,f'{args.shortcut_type}{args.percent}',split_)+'/*/*_*_*')
    assert paths != 0
    print(len(paths))
    attrs = [np.array([int(path.split('_')[-2])]) for path in paths ]
    if args.dataset.split("_")[0] == 'mimic' and args.group_type != 'no':
        with open(f"dataset/mimic/{split.replace('_flip','')}/summary.json") as load_file:
            demo_info = json.load(load_file)
        attrs = [np.array([int(path.split('_')[-2]),int(demo_info[path.split('_')[-3].split('\\')[-1].replace('-','_')][args.group_type])]) for path in paths ]
    transform_split = 'val_test' if split!='train' else 'train'
    feature, flip_feature = handle_feature(args.dataset,paths,args.ssl_ckpt_path,args.ssl_type,args.shortcut_type,args.percent,transforms[args.dataset][transform_split],split,args.use_bias_label)

    return feature, torch.from_numpy(np.asarray(attrs)), flip_feature

def concat_dummy(z):
    def hook(model, input, output):
        z.append(output)
    return hook

def get_vanilla_model(dataset,shortcut_type,shortcut_skew):

    ckpt_path = os.path.join('log',dataset,f'{shortcut_type}{shortcut_skew}_vanilla','result','best_model.th')
    print(ckpt_path)
    assert os.path.exists(ckpt_path)
    model = models.__dict__['resnet18'](num_classes=4)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.load_state_dict(torch.load(ckpt_path)['state_dict'] )

    return model

def batch_hook_feature(model,img_path_list,transforms,bz = 32):
    cnt = 0
    flag = False
    cat_feature = None
    length = len(img_path_list)
    for index in tqdm(range(length)):
        img = Image.open(img_path_list[index])
        img = transforms(img).unsqueeze(0)
        cat_img = img if cnt == 0 else torch.cat((cat_img,img),0)
        cnt += 1
        if cnt == bz or index == length - 1:
            f = []
            tmp = model(cat_img.cuda())
            hook_fn = model.avgpool.register_forward_hook(concat_dummy(f))
            _ = model(cat_img.cuda())
            hook_fn.remove()
            feature = f[0][:,:,0,0].cpu().detach().numpy()
            cat_feature = np.concatenate((cat_feature,feature),0) if flag else feature
            flag = True
            cnt = 0
    return cat_feature

def gen_hook_feature(dataset,img_path_list,shortcut_type,shortcut_skew,transforms,split):

    feature_dir = f'../LWBC/npy_files/{dataset}/vanilla/{shortcut_type}{shortcut_skew}'
    if not os.path.exists(feature_dir):
        os.makedirs(feature_dir)
    feature_fname = os.path.join(feature_dir,f'{split}_features.npy')
    flip_feature_fname = feature_fname.replace('_features','_flip_features')
    if os.path.exists(feature_fname) :
        print(feature_fname)
        return feature_fname, flip_feature_fname
    
    model = get_vanilla_model(dataset,shortcut_type,shortcut_skew).eval().cuda()
    tmp_fname = f'tmp/tmp{random.randint(0,100000)}.pth.tar'
    torch.save(model.state_dict(),tmp_fname)
    flip_size = 3
    feature = batch_hook_feature(model,img_path_list,transforms)
    np.save(feature_fname,feature)
    if split not in ['train','val_flip','test_flip'] and shortcut_type not in ['no','LO','Male','Female','Age','Race']:
        
        flip_path_list = gen_flip_path_list(img_path_list,flip_size=flip_size)
        flip_feature = np.zeros((0,feature.shape[0],feature.shape[1]))
        for i in range(flip_size):
            flip_feature_ = batch_hook_feature(model,flip_path_list[i],transforms)
            flip_feature = np.concatenate((flip_feature,np.expand_dims(flip_feature_,axis=0)))
        np.save(flip_feature_fname,flip_feature)
    else:
        flip_feature_fname = None
    state_dict = torch.load(tmp_fname)
    for k in list(state_dict.keys()):
        assert ((state_dict[k].cpu() == model.state_dict()[k].cpu()).all())
    os.remove(tmp_fname)
    return feature_fname, flip_feature_fname

def handle_feature(dataset,paths,ssl_ckpt_path,ssl_type,shortcut_type,shortcut_skew,transform,split,use_bias_label):
    dataset = dataset.replace('_ssl','')
    hook_vanilla_feature = True if ssl_ckpt_path == '' else False
    if hook_vanilla_feature:
        feature_fname,flip_feature_fname = gen_hook_feature(dataset,paths,shortcut_type,shortcut_skew,transform,split)
    else:
        feature_fname,flip_feature_fname = generate_feature(dataset,paths,ssl_ckpt_path,ssl_type,\
                                    shortcut_type,shortcut_skew,transform,split,use_bias_label)
    feature = np.load(feature_fname)
    if split!='train' and shortcut_type in IMAGE_SHORTCUT_TYPE and use_bias_label==False:
        flip_feature = np.load(flip_feature_fname)
    else:
        flip_feature = feature

    return feature, flip_feature

class COVIDDataset(Dataset):
    def __init__(self,root,split,transform,shortcut_skew,shortcut_type=None,ssl_ckpt_path='',ssl_type='',use_bias_label=False):
        super(COVIDDataset, self).__init__()
        self.split = split
        self.transform = transform
        print('Use transform ',transform)
        self.shortcut_type = shortcut_type
        split = self.split.replace('valid','val')
        self.use_bias_label = use_bias_label
        self.use_ssl = True if ssl_ckpt_path != '' else False
        if not exists(join(root,f'{shortcut_type}{shortcut_skew}')):
            covid_conversion(shortcut_type,shortcut_skew)
        self.data = glob(join(root,f'{shortcut_type}{shortcut_skew}',split)+'/*/*_*_*')
        self.length = len(self.data)
        assert self.length != 0
        if self.use_ssl:
            self.feature, self.flip_feature = handle_feature('covid',self.data,ssl_ckpt_path,ssl_type,shortcut_type,shortcut_skew,transform,self.split,use_bias_label)
            print(self.feature.shape)
        else:
            self.flip_path_list = gen_flip_path_list(self.data)


    def __len__(self):
        return self.length
    
    def gen_image(self, path):

        return self.transform(Image.open(path))

    def handle_all_single_feature(self,index):

        if self.use_ssl:
            feature = self.feature[index]
            flip_feature = self.feature[index]
            if self.split!='train' and self.shortcut_type in IMAGE_SHORTCUT_TYPE and not self.use_bias_label:
                flip_feature = self.flip_feature[:,index]
        else:
            feature = self.gen_image(self.data[index])
            if self.split!='train' and self.shortcut_type in IMAGE_SHORTCUT_TYPE and not self.use_bias_label:
                flip_feature = torch.zeros((0,feature.shape[0],feature.shape[1],feature.shape[2]))
                for i in range(3):
                    _flip_feature = self.gen_image(self.flip_path_list[i][index])
                    flip_feature = torch.cat((flip_feature,_flip_feature.unsqueeze(0)))        
            else:
                #flip_feature = torch.zeros_like(feature)
                flip_feature = torch.zeros(1,1)

        return feature,flip_feature
    
    def __getitem__(self, index):
        label = int(self.data[index].split('_')[-2])
        bias_label = int(self.data[index].split('_')[-1].split('.')[0])
        feature, flip_feature = self.handle_all_single_feature(index)
        attr = torch.LongTensor([label,bias_label])
        return feature, attr, flip_feature
    
transforms = {
    "covid": {
        "train": T.Compose([T.Grayscale(num_output_channels=1),T.ToTensor()]),
        "val_test": T.Compose([T.Grayscale(num_output_channels=1),T.ToTensor()]),
        },
    "covid_ssl": {
        "train": T.Compose([T.Grayscale(num_output_channels=1),T.ToTensor()]),
        "val_test": T.Compose([T.Grayscale(num_output_channels=1),T.ToTensor()]),
        },
    "mimic": {
        "train": T.Compose([T.Resize((256,256)),T.Grayscale(num_output_channels=1),T.ToTensor()]),
        "val_test": T.Compose([T.Resize((256,256)),T.Grayscale(num_output_channels=1),T.ToTensor()])
        },
    "mimic_ssl": {
        "train": T.Compose([T.Resize((256,256)),T.Grayscale(num_output_channels=1),T.ToTensor()]),
        "val_test": T.Compose([T.Resize((256,256)),T.Grayscale(num_output_channels=1),T.ToTensor()])
        },

    }
class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x
transforms_preprcs = {
    "mimic": {
        "train": T.Compose([T.Grayscale(num_output_channels=1),
                            T.RandomResizedCrop((256,256),scale=(0.682,0.77)),
                            T.RandomApply([
                                T.ColorJitter(0.2, 0.2, 0.2, 0.1)  # not strengthened
                            ], p=0.5),
                            T.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
                            T.ToTensor()]),
        "val_test": T.Compose([T.Resize((256,256)),
                               T.Grayscale(num_output_channels=1),
                            T.ToTensor()
                            ])
        },
    "covid": {
        "train": T.Compose([T.Grayscale(num_output_channels=1),
                            T.RandomResizedCrop((299,299),scale=(0.8,0.9)),
                            T.RandomApply([
                                T.ColorJitter(0.2, 0.2, 0.2, 0.1)  # not strengthened
                            ], p=0.5),
                            T.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
                            T.ToTensor()]),
        "val_test": T.Compose([T.Grayscale(num_output_channels=1),
                            T.ToTensor()]),
        },
    
}

def get_dataset(args, dataset_split, transform_split, use_preprocess=None):
    dataset = args.dataset
    percent = float(args.percent)
    shortcut_type = args.shortcut_type
    group_type = args.group_type
    ssl_ckpt_path = args.ssl_ckpt_path
    ssl_type = args.ssl_type
    if use_preprocess:
        transform = transforms_preprcs[dataset][transform_split]
    else:
        transform = transforms[dataset][transform_split]

    dataset_split = "valid" if (dataset_split == "val") else dataset_split
    if dataset in ['covid','covid_ssl']:
        root = 'dataset/COVID/'
        assert shortcut_type not in ['LO','Male','Female','Age','Race']
        dataset = COVIDDataset(root=root, split=dataset_split, shortcut_skew = percent, shortcut_type=shortcut_type, transform = transform, ssl_type=ssl_type, ssl_ckpt_path=ssl_ckpt_path ,use_bias_label=args.use_bias_label)
    elif dataset in ['mimic','mimic_ssl']:
        root = f"dataset/MIMIC"
        dataset = MimicDataset(root=root, split=dataset_split, shortcut_skew = percent, shortcut_type=shortcut_type, group_type=group_type, transform = transform, ssl_type=ssl_type, ssl_ckpt_path=ssl_ckpt_path ,use_bias_label=args.use_bias_label)
    else:
        print('wrong dataset ...')
        import sys
        sys.exit(0)

    return dataset

def write_out_scalar(self,scalar_dict,step):
    if self.args.tensorboard:
        for key in scalar_dict.keys():
            self.writer.add_scalar(key,scalar_dict[key],step)
    if self.args.wandb:
        wandb.log(scalar_dict,step=step)

def get_class_info(dataset,shortcut_type,group_type,use_bias_label):

    if use_bias_label:
        if shortcut_type in ['mark','lightness','contrast','jpeg']:
            return [f'{shortcut_type} {i+1}' for i in range(4)],shortcut_type.lower(),4
        elif shortcut_type == 'LO':
            return ['MIMIC','COVID'],'dataset',2
        elif group_type == 'gender':
            return ['Male','Female'],'gender',2
        elif group_type == 'age':
            return ['0~19y','20~39y','40~59y','60~79y','80y~'],'age',5
        elif group_type == 'race':
            return ['White','Black','Asian'],'race',3
        else:
            raise NotImplementedError
    elif dataset in ['mimic','mimic_ssl']:
        return ['No Finding','Lung Opacity','Cardiomegaly','Pleural Effusion'],'disease',4
    elif dataset in ['covid','covid_ssl']:
        return ['Normal','COVID','Lung_opacity','Viral Pneumonia'],'disease',4
    else:
        raise NotImplementedError
  
def write_out_cm(self,cm_name,label,pred):
    if self.args.wandb:
        if self.args.dataset in ['bar_ssl','bar']:
            cm = plot.confusion_matrix(
                y_true=label.tolist(),
                preds=pred.tolist())
        
        else:

            class_name,target_name,class_num = get_class_info(self.args.dataset,self.args.shortcut_type,self.args.group_type,self.args.use_bias_label)

            cm = plot.confusion_matrix(
                y_true=label.tolist(),
                preds=pred.tolist(),
                class_names=class_name)
        wandb.log({cm_name: cm})