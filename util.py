from tqdm import tqdm
import torch
import numpy as np
import torch.nn as nn
import os
from sklearn.metrics import confusion_matrix, roc_auc_score
from itertools import combinations
import json
from pathlib import Path
import wandb
from wandb import plot

softmax = nn.Softmax(dim=1)

GROUP_LIST = ['age', 'race', 'gender']
GROUP_LIST_ = ['age', 'race', 'gender', 'LO']
METHOD_LIST = ['', 'knn_', 'xgb_']
BIAS_LABEL_LIST = ['', 'mark_', 'lightness_', 'contrast_', 'jpeg_', 'age_', 'gender_', 'race_']
MATRIX_LIST = ['val_acc', 'test_acc', 'val_auc', 'test_auc', 'test_auc_aprart']
RESULT_ITEM = (
    [f'{method}{bias_label}{item}' for item in MATRIX_LIST for bias_label in BIAS_LABEL_LIST for method in METHOD_LIST] +
    [f'{method}val_fair_group_{group}' for group in GROUP_LIST_ for method in METHOD_LIST] +
    [f'{method}test_fair_group_{group}' for group in GROUP_LIST_ for method in METHOD_LIST] +
    [f'{method}val_pair_info_group_{group}' for group in GROUP_LIST for method in METHOD_LIST] +
    [f'{method}test_pair_info_group_{group}' for group in GROUP_LIST for method in METHOD_LIST]
)

def join_paths(*kwawrg):
    return os.path.join(*kwawrg)

def path_exists(*paths):
    return os.path.exists(join_paths(*paths))

def load_json_file(path):
    """讀取 JSON 文件"""
    with open(path, "r") as infile:
        return json.load(infile)

def save_json_file(data, path):
    """保存數據到 JSON 文件"""
    with open(path, "w") as outfile:
        json.dump(data, outfile, indent=4, sort_keys=True)

def get_feature_num(ssl_type, ssl_ckpt_path, use_bias_label):
    feature_dimensions = {
        'simsiam': 2048,
        'byol': {'resnet18': 512},
        'simclr': {'resnet18': 512, 'resnet50': 2048},
        'dino': {'vit_tiny': 2048, 'vit_small': 384, 'vit_base': 2048, 'resnet18': 512, 'resnet50': 2048},
        'swav': {'resnet18': 2048, 'resnet50': 2048},
    }

    if ssl_type in feature_dimensions:
        arch = ssl_ckpt_path.split('/')[-2].split('_')[0]
        return feature_dimensions[ssl_type].get(arch, 0)
    elif use_bias_label:
        return 512
    return 0

def calculate_tpr_fpr(label,pred):
    """計算 TPR 和 FPR"""
    tn, fp, fn, tp = confusion_matrix(label, pred, labels=[0, 1]).ravel()
    tpr = tp / (tp + fn) if (tp + fn) != 0 else 0.0
    fpr = fp / (tn + fp) if (tn + fp) != 0 else 0.0
    return tpr,fpr

def cal_img_bias_fairness(labels,preds,flip_preds,num_class,flip_size):

    fairness = 0.
    for i in range(num_class):
        tpr,fpr = calculate_tpr_fpr(labels==i,preds==i)     
        for j in range(flip_size):
            flip_tpr,flip_fpr = calculate_tpr_fpr(labels==i,flip_preds[j]==i)
            fairness += ( abs(flip_tpr-tpr) + abs(flip_fpr-fpr) )
    return fairness / 2 / flip_size / num_class

def cal_demo_group_fairness(labels,preds,group):
    num_class = int(torch.max(labels).item()) + 1
    num_group = int(torch.max(group).item()) + 1

    fairness = 0.
    
    group_labels = [ labels[group==i] for i in range(num_group) ]
    group_preds = [ preds[group==i] for i in range(num_group) ]
        
    group_tpr_fpr_dict = dict()
    for group_idx in range(num_group):
        tpr_list, fpr_list = [],[]
        for class_idx in range(num_class):
            tpr,fpr = calculate_tpr_fpr(group_labels[group_idx]==class_idx,group_preds[group_idx]==class_idx)
            tpr_list.append(tpr)
            fpr_list.append(fpr)
        group_tpr_fpr_dict[group_idx] = dict()
        group_tpr_fpr_dict[group_idx]['tpr'] = tpr_list
        group_tpr_fpr_dict[group_idx]['fpr'] = fpr_list
    max_fairness = 0
    pair_info = dict()
    for g1,g2 in list(combinations(range(num_group), 2)):

        fairness = 0
        for key in ['tpr','fpr']:
            for class_idx in range(num_class):
                fairness += ( abs(group_tpr_fpr_dict[g1][key][class_idx]-group_tpr_fpr_dict[g2][key][class_idx]))
        fairness = fairness / 2 / num_class

        pair_info[f'{g1}<-->{g2}'] = fairness
        if fairness > max_fairness:
            max_fairness = fairness
    return max_fairness, pair_info

def cal_label_bias_fairness_fit(labels,preds,flip_labels,flip_preds):
    LO_CLASS = 1
    tpr1,fpr1 = calculate_tpr_fpr(flip_labels==LO_CLASS,flip_preds==LO_CLASS)
    tpr2,fpr2 = calculate_tpr_fpr(labels==LO_CLASS,preds==LO_CLASS)

    return (abs(tpr1-tpr2)+abs(fpr1-fpr2))/2

def cal_label_bias_fairness_cnn(labels,preds,LO_flip_dataloader,model,device):

    fairness_list = []
    LO_flip_pred, LO_flip_label = torch.tensor([]), torch.tensor([])
    for data, attr, _ in tqdm(LO_flip_dataloader, leave=False):
        data = data.to(device)
        with torch.no_grad():
            pred = model(data).data.max(1, keepdim=True)[1].squeeze(1)
            LO_flip_label = torch.cat((LO_flip_label,attr[:, 0]))
            LO_flip_pred = torch.cat((LO_flip_pred,pred.detach().cpu()))
    for i in range(4):
        tpr1,fpr1 = calculate_tpr_fpr(LO_flip_label==i,LO_flip_pred==i)
        tpr2,fpr2 = calculate_tpr_fpr(labels==i,preds==i)
        fairness_list.append((abs(tpr1-tpr2)+abs(fpr1-fpr2))/2)
    avg_fairness = sum(fairness_list) / 4
    fairness_list.append(avg_fairness)

    return fairness_list

def cal_demo_bias_demo_group_fairness_cnn(origin_loader,model,device):

    preds, labels, attrs = torch.tensor([]), torch.tensor([]), torch.tensor([])
    for data, attr, _ in tqdm(origin_loader, leave=False):
        data = data.to(device)
        with torch.no_grad():
            pred = model(data).data.max(1, keepdim=True)[1].squeeze(1)
            labels = torch.cat((labels,attr[:, 0]))
            preds = torch.cat((preds,pred.detach().cpu()))
            attrs = torch.cat((attrs,attr[:,1]))
  
    return  cal_demo_group_fairness(labels,preds,attrs)

def get_old_result(result_dir):

    result_path = join_paths(result_dir, "result.json")
    if path_exists(result_path):
        # 使用 `with` 來安全地打開文件
        result = load_json_file(result_path)
        # 僅保留 RESULT_ITEM 列表中的鍵
        result = {key: value for key, value in result.items() if key in RESULT_ITEM}
    else:
        result = {}

    return result

def save_result(result,result_dir):

    result_path = join_paths(result_dir, "result.json")
    with open(result_path, "w") as outfile:
        json.dump(result, outfile,indent=4,sort_keys=True)

def torch_safe_save(obj, path):
    """確保存檔時路徑存在"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)  # 建立資料夾（含父層）
    torch.save(obj, path)

def get_device(args=None):
    if args and getattr(args, "device", None):
        return torch.device(args.device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def safe_roc_auc_score(y_true, y_score, default=0.5, **kwargs):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    # 檢查輸入數據
    if len(np.unique(y_true)) < 2 or np.all(y_score == y_score[0]):
        return default

    if np.any(np.isnan(y_true)) or np.any(np.isnan(y_score)) or np.any(np.isinf(y_true)) or np.any(np.isinf(y_score)):
        return default

    try:
        return roc_auc_score(y_true, y_score, **kwargs)
    except:
        return default

def check_retrain(log_path, continue_train):
    if path_exists(log_path) and not continue_train:
        user_input = input(f'Do you want to re-train model in {log_path} (yes/no) :')
        if user_input.lower() in ['yes', 'y']:
            pass
        elif user_input.lower() in ['no', 'n']:
            exit()
        else:
            print('Type yes or no')
            exit()

def get_class_info(dataset, shortcut_type, group_type, use_bias_label):
    group_info = {
        'mark': ([f'{shortcut_type} {i+1}' for i in range(4)], 'mark', 4),
        'lightness': ([f'{shortcut_type} {i+1}' for i in range(4)], 'lightness', 4),
        'contrast': ([f'{shortcut_type} {i+1}' for i in range(4)], 'contrast', 4),
        'jpeg': ([f'{shortcut_type} {i+1}' for i in range(4)], 'jpeg', 4),
        'LO': (['MIMIC', 'COVID'], 'dataset', 2),
        'gender': (['Male', 'Female'], 'gender', 2),
        'age': (['0~19y', '20~39y', '40~59y', '60~79y', '80y~'], 'age', 5),
        'race': (['White', 'Black', 'Asian'], 'race', 3),
    }

    if use_bias_label and shortcut_type in group_info:
        return group_info[shortcut_type]
    elif dataset in ['mimic', 'mimic_ssl']:
        return ['No Finding', 'Lung Opacity', 'Cardiomegaly', 'Pleural Effusion'], 'disease', 4
    elif dataset in ['covid', 'covid_ssl']:
        return ['Normal', 'COVID', 'Lung_opacity', 'Viral Pneumonia'], 'disease', 4
    else:
        raise NotImplementedError

def write_out_cm(self, cm_name, label, pred):
    if self.args.wandb:
        class_name, target_name, class_num = get_class_info(
            self.args.dataset, self.args.shortcut_type, self.args.group_type, self.args.use_bias_label
        )
        cm = plot.confusion_matrix(
            y_true=label.tolist(),
            preds=pred.tolist(),
            class_names=class_name if self.args.dataset not in ['bar_ssl', 'bar'] else None
        )
        wandb.log({cm_name: cm})

def write_out_scalar(self,scalar_dict,step):
    if self.args.tensorboard:
        for key in scalar_dict.keys():
            self.writer.add_scalar(key,scalar_dict[key],step)
    if self.args.wandb:
        wandb.log(scalar_dict,step=step)