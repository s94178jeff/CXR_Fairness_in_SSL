from tqdm import tqdm
import wandb
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import json
import os
from util import join,cal_img_bias_fairness,cal_demo_group_fairness,cal_label_bias_fairness_cnn,cal_label_bias_fairness_fit,\
    cal_demo_bias_demo_group_fairness_cnn,get_old_result,save_result,check_retrain,get_feature_num
import torch.optim as optim
from data_util import get_dataset,get_dataset_feature, IdxDataset,IMAGE_SHORTCUT_TYPE
from module.util import get_model
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
import random
from importlib.machinery import SourceFileLoader
from ssl_inference import get_ssl_info
softmax = nn.Softmax(dim=1)
from util import GROUP_LIST_,GROUP_LIST,METHOD_LIST,MATRIX_LIST,BIAS_LABEL_LIST,RESULT_ITEM, torch_safe_save, get_device, safe_roc_auc_score, get_class_info\
    ,write_out_scalar, write_out_cm
from pathlib import Path

class Learner(object):
    def __init__(self, args):

        data2model = {
                       'covid':"ResNet18",
                       'mimic':"ResNet18",
                       'mimic_ssl': "1linear",
                       'covid_ssl': "1linear"}

        data2batch_size = {'covid':256,
                           'mimic':512,
                           'mimic_ssl':512,
                           'covid_ssl':512}
        
        data2preprocess = {
                           'mimic': False,
                           'covid': False,
                           'mimic_ssl':None,
                           'covid_ssl':None
                           }
        if args.method == 'aug_vanilla':
            data2preprocess['mimic'] = True
            data2preprocess['covid'] = True
        run_name = f'{args.shortcut_type}{args.percent}_{args.method}'
        if args.ssl_type!='':
            aug, epoch = get_ssl_info(args.ssl_ckpt_path)
            run_name = f'{args.ssl_type}_{args.shortcut_type}{args.percent}_ep{epoch}_{aug}_{args.method}'
        if args.exp != '':
            run_name = args.exp
        self.run_name = run_name

        args.wandb = False if args.local else True
        args.tensorboard = False
        
        self.model = data2model[args.dataset]
        self.batch_size = data2batch_size[args.dataset]
        if args.model != '':
            self.model = args.model
        args.model = self.model
        run_name_title = run_name+'(bias)' if args.use_bias_label else run_name
        if args.use_bias_label:
            if args.shortcut_type in IMAGE_SHORTCUT_TYPE+['LO']:
                run_name_title = f'{run_name}{args.shortcut_type}'
            elif args.group_type in GROUP_LIST:
                run_name_title = f'{run_name}{args.group_type}'
            else:
                raise NotImplementedError
        if args.wandb:
            import wandb
            wandb.init(project=f'LDD_{args.dataset}', resume=args.continue_train,config=args)
            wandb.run.name = run_name_title
        
        if args.tensorboard:
            from tensorboardX import SummaryWriter
            self.writer = SummaryWriter(f'result/summary/{run_name_title}')
        print(f'bias model: {self.model} || dataset: {args.dataset}')
        print(f'working with experiment: {run_name_title}...')

        self.result_root = os.makedirs(join(args.result_root, args.dataset, run_name), exist_ok=True)
        self.device = get_device(args)
        print(f"Using device: {self.device}")
        self.args = args

        print(self.args)

        # logging directories
        self.result_root = join(args.result_root, args.dataset, run_name)
        use_bias_str = 'bias_label_' if args.use_bias_label else ''
        self.result_dir = join(self.result_root, f"{use_bias_str}result")
        if not args.use_bias_label or args.group_type not in ['age','race','gender']:
            check_retrain(self.result_dir,args.continue_train)
        os.makedirs(self.result_dir, exist_ok=True)

        feature_num = get_feature_num(args.ssl_type,args.ssl_ckpt_path,args.use_bias_label)
        if feature_num:
            print('feature num ',feature_num)
        self.num_channel = 1 if args.dataset in ['mimic','covid'] else 3
        self.attr_idx = 1 if args.use_bias_label else 0
        self.valid_dataset = get_dataset(
            args,
            dataset_split="valid",
            transform_split="val_test",
            use_preprocess=data2preprocess[args.dataset],
        )
        self.train_dataset = get_dataset(
            args,
            dataset_split="train",
            transform_split="train",
            use_preprocess=data2preprocess[args.dataset],
        )
        self.test_dataset = get_dataset(
            args,
            dataset_split="test",
            transform_split="val_test",
            use_preprocess=data2preprocess[args.dataset],
        )
        _,self.target_name,self.num_classes = get_class_info(args.dataset,args.shortcut_type,args.group_type,args.use_bias_label)

        self.train_dataset = IdxDataset(self.train_dataset)
        pin_memory = True if self.device.type == 'cuda' else False
        # make loader
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
            drop_last=False
        )

        self.valid_loader = DataLoader(
            self.valid_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=pin_memory
        )

        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=pin_memory
        )
        
        if (args.shortcut_type in ['LO','Male','Female','Race','Age'] or args.group_type == 'LO') and args.dataset not in ['covid','covid_ssl']:
            
            self.valid_flip_dataset = get_dataset(
                args,
                dataset_split="valid_flip",
                transform_split="val_test",
                use_preprocess=data2preprocess[args.dataset],
            )
            
            self.test_flip_dataset = get_dataset(
                args,
                dataset_split="test_flip",
                transform_split="val_test",
                use_preprocess=data2preprocess[args.dataset],
            )
            
            self.valid_flip_loader = DataLoader(
                self.valid_flip_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=True
            )

            self.test_flip_loader = DataLoader(
                self.test_flip_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=True
            )

        self.warm_up_cnt = 0
        self.warm_up = False

        # define model and optimizer
        self.model = get_model(self.model, self.num_classes, self.num_channel,ssl_feature=feature_num).to(self.device)
        init_lr = args.lr * self.batch_size / 256
        print('lr: ' ,init_lr)
        self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=init_lr,
                weight_decay=args.weight_decay,
            )

        # define loss
        self.criterion = nn.CrossEntropyLoss(reduction='none')

        print(f'self.criterion: {self.criterion}')

        self.best_valid_acc, self.best_test_acc = 0., 0.
        self.best_val_auc, self.best_test_auc = 0., 0.
        print('finished model initialization....')

    def evaluate(self, model, data_loader):
        model.eval()
        total_num = 0
        cat_pred, cat_prob, cat_attr = torch.tensor([]), torch.tensor([]), torch.tensor([])
        for data, attr, _ in tqdm(data_loader, leave=False):
            data = data.to(self.device)

            with torch.no_grad():

                logit = model(data)
                pred = logit.data.max(1, keepdim=True)[1].squeeze(1)
                cat_prob = torch.cat((cat_prob,softmax(logit).detach().cpu()))
                cat_attr = torch.cat((cat_attr,attr[:, self.attr_idx]))
                cat_pred = torch.cat((cat_pred,pred.detach().cpu()))
                total_num += attr.shape[0]

        accs = (cat_attr==cat_pred).long().sum().item()/float(total_num)
        model.train()

        return accs , (cat_attr,cat_pred,cat_prob)

    # evaluation code for vanilla
    def evaluate_fair(self, model, data_loader,shortcut_type='no',group_type='no',flip_dataloader=None,split='val'):
        model.eval()
        total_num = 0
        cat_pred, cat_label, cat_prob, cat_attr, cat_flip_pred = torch.tensor([]), torch.tensor([]), torch.tensor([]), torch.tensor([]), torch.tensor([])

        for data, attr, flip_data in tqdm(data_loader, leave=False):
            data = data.to(self.device)
            flip_data = flip_data.to(self.device).to(torch.float32)

            with torch.no_grad():

                logit = model(data)
                pred = logit.data.max(1, keepdim=True)[1].squeeze(1)
                if shortcut_type in IMAGE_SHORTCUT_TYPE:
                    flip_pred = torch.tensor([])
                    for i in range(flip_data.shape[1]):
                        _flip_pred = model(flip_data[:,i]).data.max(1, keepdim=True)[1].squeeze(1)
                        flip_pred = torch.cat((flip_pred,_flip_pred.unsqueeze(0).detach().cpu()))
                    cat_flip_pred = torch.cat((cat_flip_pred,flip_pred),dim=1)
                if group_type in GROUP_LIST:
                    cat_attr = torch.cat((cat_attr,attr[:,1]))
                cat_prob = torch.cat((cat_prob,softmax(logit).detach().cpu()))
                cat_label = torch.cat((cat_label,attr[:, 0]))
                cat_pred = torch.cat((cat_pred,pred.detach().cpu()))
                total_num += attr.shape[0]

        accs = (cat_label==cat_pred).long().sum().item()/float(total_num)
        pair_info = None
        if shortcut_type in IMAGE_SHORTCUT_TYPE:
            fairness = cal_img_bias_fairness(labels=cat_label,preds=cat_pred,flip_preds=cat_flip_pred,num_class=4,flip_size=3)
        elif shortcut_type == 'no' and group_type in GROUP_LIST:
            assert self.args.dataset not in ['covid','covid_ssl']
            fairness,pair_info = cal_demo_group_fairness(labels=cat_label,preds=cat_pred,group=cat_attr)
        elif shortcut_type == 'LO' or group_type == 'LO':
            fairness = cal_label_bias_fairness_cnn(cat_label,cat_pred,flip_dataloader,model,self.device)
        elif shortcut_type in ['Male','Female','Race','Age']:
            fairness,pair_info = cal_demo_bias_demo_group_fairness_cnn(flip_dataloader,model,self.device)
            print(fairness)
            print(pair_info)
        else:
            fairness = -1
            print(shortcut_type)
            raise NotImplementedError
        model.train()

        return accs ,fairness, pair_info , (cat_label,cat_pred,cat_prob)

    def load_vanilla(self, best=None):
        if self.args.use_bias_label and self.args.group_type in ['age','race','gender']:
            bias_str = self.args.group_type+'_'
        else:
            bias_str = ''
        if best:
            model_path = join(self.result_dir, f"{bias_str}best_model.th")
        else:
            model_path = join(self.result_dir, f"{bias_str}final_model.th")
        self.model.load_state_dict(torch.load(model_path)['state_dict'])
        self.optimizer.load_state_dict(torch.load(model_path)['optimizer'])
        return torch.load(model_path)['steps']

    def save_vanilla(self, step, best=None):
        if self.args.use_bias_label and self.args.group_type in ['age','race','gender']:
            bias_str = self.args.group_type+'_'
        else:
            bias_str = ''
        if best:
            model_path = join(self.result_dir, f"{bias_str}best_model.th")
        else:
            model_path = join(self.result_dir, f"{bias_str}final_model.th")
        state_dict = {
            'steps': step,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        if best:
            state_dict[f'best_valid_acc'] = self.best_valid_acc
            state_dict[f'best_test_acc'] = self.best_test_acc
        torch_safe_save(state_dict,model_path)
        print(f'{step} model saved ...')

    def board_vanilla_loss(self, step, loss):
        
        write_out_scalar(self,{
                "loss/loss_train":loss
            },step)

    def board_shorTest_acc(self, step, epoch, inference=None):
        shortcut = self.args.shortcut_type
        group = self.args.group_type
        val_accs, val_auc = 0, 0
        val_cat_label, val_cat_pred = torch.tensor([]),torch.tensor([])
        val_pair_info, test_pair_info = dict(),dict()

        if shortcut in ['LO','Male','Female','Race','Age'] or group == 'LO':
            val_accs,val_fair,_,(val_cat_label,val_cat_pred,val_cat_prob) = self.evaluate_fair(self.model, self.valid_loader,shortcut,group,self.valid_flip_loader)
        else:
            val_accs,val_fair,val_pair_info,(val_cat_label,val_cat_pred,val_cat_prob) = self.evaluate_fair(self.model, self.valid_loader,shortcut,group)
        val_auc = safe_roc_auc_score(val_cat_label, val_cat_prob,multi_class='ovr',labels=[0,1,2,3])
        result = get_old_result(self.result_dir)
        
        result['val_acc'] = val_accs
        result['val_auc'] = val_auc
        result[f'val_fair_group_{group}'] = val_fair
        print('result',result)
        if group in GROUP_LIST:
            result[f'val_pair_info_group_{group}'] = val_pair_info
            result[f'test_pair_info_group_{group}'] = test_pair_info
        if inference:
            if shortcut in ['LO','Male','Female','Race','Age'] or group == 'LO':
                test_accs,test_fair,_,(test_cat_label,_,test_cat_prob) = self.evaluate_fair(self.model, self.test_loader,shortcut,group,self.test_flip_loader,split='test')
            else:
                test_accs,test_fair,test_pair_info,(test_cat_label,_,test_cat_prob) = self.evaluate_fair(self.model, self.test_loader,shortcut,group,split='test')
            test_auc = safe_roc_auc_score(test_cat_label, test_cat_prob,multi_class='ovr',labels=[0,1,2,3])
            if shortcut == 'LO' or group == 'LO':
                test_auc_list = []
                for class_idx in range(4):
                    test_auc_list.append(safe_roc_auc_score(test_cat_label==class_idx, test_cat_prob[:,class_idx]))
                result['test_auc_apart'] = test_auc_list
            result['test_acc'] = test_accs
            result['test_auc'] = test_auc
            result[f'test_fair_group_{group}'] = test_fair
            if group in GROUP_LIST:
                result[f'test_pair_info_group_{group}'] = test_pair_info
            save_result(result,self.result_dir)
            print(f'evaluation done')
            import sys
            sys.exit(0)
        print(f'epoch: {epoch}')

        write_out_scalar(self,{
                "acc/valid": val_accs,
                "acc/best_valid": self.best_valid_acc,
                "auc/valid": val_auc,
                "auc/best_valid": self.best_val_auc,
            },step)
        #metric val auc
        if val_auc >= self.best_val_auc:
            self.best_val_auc = val_auc
            self.best_valid_acc = val_accs
            write_out_scalar(self,{
                f"fairness_{group}/val":val_fair,
            },step)
            write_out_cm(self,"best_val_conf_mat",val_cat_label,val_cat_pred) 
            save_result(result,self.result_dir)
            self.save_vanilla(step, best=True)

        print(f'val acc: {val_accs} || val auc: {val_auc}')
    
    def board_vanilla_acc(self, step, epoch, inference=None):
        val_accs,(val_attr,val_pred,val_prob) = self.evaluate(self.model, self.valid_loader)
        if self.args.use_bias_label:
            tn = self.target_name + '_'
        else:
            tn = ''
        if self.num_classes == 2 :
            val_auc = safe_roc_auc_score(val_attr, val_prob[:,1])
        else:
            val_auc = safe_roc_auc_score(val_attr, val_prob,multi_class='ovr',labels=list(range(self.num_classes)))
        result = get_old_result(self.result_dir)
        result[f'{tn}val_acc'] = val_accs
        result[f'{tn}val_auc'] = val_auc
        if inference:
            if self.args.shortcut_type=='LO':
                test_accs,(test_attr,test_pred,test_prob) = self.evaluate(self.model, self.test_flip_loader)
            else:
                test_accs,(test_attr,test_pred,test_prob) = self.evaluate(self.model, self.test_loader)
            if self.num_classes == 2 :
                test_auc = safe_roc_auc_score(test_attr, test_prob[:,1])
            else:
                test_auc = safe_roc_auc_score(test_attr, test_prob,multi_class='ovr',labels=list(range(self.num_classes)))
            result[f'{tn}test_acc'] = test_accs
            result[f'{tn}test_auc'] = test_auc
            save_result(result,self.result_dir)
            print(f'evaluation done')
            import sys
            sys.exit(0)
        print(f'epoch: {epoch}')
        write_out_scalar(self,{
                f"acc/{tn}valid": val_accs,
                f"acc/{tn}best_valid": self.best_valid_acc,
                f"auc/{tn}valid": val_auc,
                f"auc/{tn}best_valid": self.best_val_auc,
            },step)
        #metric val auc
        if val_auc >= self.best_val_auc or self.best_val_auc==None:
            self.best_val_auc = val_auc
            self.best_valid_acc = val_accs
            write_out_cm(self,f"best_{tn}val_conf_mat",val_attr,val_pred) 
            save_result(result,self.result_dir)
            self.save_vanilla(step, best=True)

        print(f'val acc: {val_accs} || val auc: {val_auc}')

    def sanity_check(self,state_dict, pretrained_weights):
        """
        Linear classifier should not change any weights other than the linear layer.
        This sanity check asserts nothing wrong happens (e.g., BN stats updated).
        """
        #print("=> loading '{}' for sanity check".format(pretrained_weights))
        checkpoint = torch.load(pretrained_weights, map_location="cpu")
        state_dict_pre = checkpoint['state_dict']
        change_key = []

        for k in list(state_dict.keys()):
            # only ignore fc layer
            #if 'fc.weight' in k or 'fc.bias' in k:
            #    print(k)
            #    assert not ((state_dict[k].cpu() == state_dict_pre[k].cpu()).all())
            #print(k)
            #k = k[len("module."):]
            #print(k)

            # name in pretrained model
            
            k_pre = 'encoder.' + k[len('module.'):] \
                if k.startswith('module.') else 'encoder.' + k
            #print(k,k_pre)
            if not (state_dict[k].cpu() == state_dict_pre[k]).all():
                change_key.append(k)
        print('{} is changed in training.'.format(change_key))
        os.remove(pretrained_weights)
        #print("=> sanity check passed.")
    
    def reg_evaluation(self,split,model_name,features,attrs,flip_features,flip_attrs):
        group = self.args.group_type
        shortcut_type = self.args.shortcut_type
        labels = attrs[:,self.attr_idx]
        if self.args.use_bias_label:
            tn = self.target_name + '_'
        else:
            tn = ''
        if model_name == 'knn':
            model = self.knn
        elif model_name == 'xgb':
            model = self.xgb
        else:
            model = self.knn
        pred = model.predict(features)
        prob = model.predict_proba(features)
        #auc = safe_roc_auc_score(labels, prob,multi_class='ovr',labels=[0,1,2,3])
        if self.num_classes == 2 :
            auc = safe_roc_auc_score(labels, prob[:,1])
        else:
            auc = safe_roc_auc_score(labels, prob,multi_class='ovr',labels=list(range(self.num_classes)))
        acc = model.score(features, labels)
        self.result[f'{model_name}_{tn}{split}_acc'] = acc
        self.result[f'{model_name}_{tn}{split}_auc'] = auc
        if self.args.shorTest:
            if shortcut_type in IMAGE_SHORTCUT_TYPE:
                flip_size = flip_features.shape[0]
                flip_pred = []
                for i in range(flip_size):
                    flip_pred.append(model.predict(flip_features[i]))
                fairness = cal_img_bias_fairness(labels=labels,preds=pred,flip_preds=flip_pred,num_class=self.num_classes,flip_size=flip_size)
            elif shortcut_type == 'no' and group in GROUP_LIST:
                fairness,pair_info = cal_demo_group_fairness(labels=labels,preds=pred,group=attrs[:,1])
            elif shortcut_type in ['LO']:
                flip_pred = model.predict(flip_features)
                fairness = cal_label_bias_fairness_fit(labels,pred,flip_attrs[:,0],flip_pred)
            elif shortcut_type in ['Male','Female','Race','Age']:
                flip_pred = model.predict(flip_features)
                fairness,pair_info = cal_demo_group_fairness(flip_attrs[:,0],flip_pred,flip_attrs[:,1])
                print(fairness)
                print(pair_info)
            else:
                fairness = -1
            self.result[f'{model_name}_{split}_fair_group_{group}'] = fairness
            if group in GROUP_LIST:
                self.result[f'{model_name}_{split}_pair_info_group_{group}'] = pair_info

    def fit_other(self, args):
        assert args.dataset in ['mimic_ssl','covid_ssl']

        self.result = get_old_result(self.result_dir)

        #if args.shortcut_type in IMAGE_SHORTCUT_TYPE:
        group = args.group_type
        train_features, train_attr, _ = get_dataset_feature(args,'train')
        val_features, val_attrs, flip_val_features = get_dataset_feature(args,'valid')
        test_features, test_attrs, flip_test_features = get_dataset_feature(args,'test')
        if args.shortcut_type in ['LO','Male','Female','Race','Age']:
            flip_val_features, flip_val_attrs, _ = get_dataset_feature(args,'valid_flip')
            flip_test_features, flip_test_attrs, _ = get_dataset_feature(args,'test_flip')
        else:
            flip_val_attrs, flip_test_attrs = None,None
        self.knn = KNeighborsClassifier()
        self.knn.fit(train_features, train_attr[:,self.attr_idx])
        #self.xgb = XGBClassifier()
        #self.xgb.fit(train_features, train_attr[:,self.attr_idx])
        for model_name in ['knn']:
        #for model_name in ['knn','xgb']:
           self.reg_evaluation('val',model_name,val_features,val_attrs,flip_val_features,flip_val_attrs)
           self.reg_evaluation('test',model_name,test_features,test_attrs,flip_test_features,flip_test_attrs)
        save_result(self.result,self.result_dir)
        
    def train_vanilla(self, args):
        start_step = self.load_vanilla() if args.continue_train else 0

        train_iter = iter(self.train_loader)
        train_num = len(self.train_dataset.dataset)
        epoch, cnt = 0, 0
        tmp_fname = Path("tmp") / f'tmp{random.randint(0,100000)}.pth.tar'
        torch_safe_save({'state_dict': self.model.state_dict()}, tmp_fname)
        for step in tqdm(range(start_step,args.num_steps), leave=False):
            self.model.train()
            try:
                index, data, attr, _ = next(train_iter)
            except:
                train_iter = iter(self.train_loader)
                index, data, attr, _ = next(train_iter)

            data = data.to(self.device)
            logit = self.model(data)
            attr = attr.to(self.device)
            
            label = attr[:,self.attr_idx]
            loss_update = self.criterion(logit, label)
            loss = loss_update.mean()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            ##################################################
            #################### LOGGING #####################
            ##################################################

            if step == 0:
                self.sanity_check(self.model.state_dict(),tmp_fname)

            if step % args.save_freq == 0:
                self.save_vanilla(step)

            if step % args.log_freq == 0:
                self.board_vanilla_loss(step, loss=loss)

            if step % args.valid_freq == 0:
                if args.shorTest:
                    self.board_shorTest_acc(step,epoch)
                else:
                    self.board_vanilla_acc(step, epoch)
            cnt += len(index)
            if cnt == train_num:
                print(f'finished epoch: {epoch}')
                epoch += 1
                cnt = 0
            if step == args.num_steps - 1:
                self.test_vanilla(args)

    def test_vanilla(self, args):
        self.load_vanilla(best=True)
        #model_path = join(self.result_dir, 'best_model.th')
        #self.model.load_state_dict(torch.load(model_path)['state_dict'])
        if args.shorTest:
            self.board_shorTest_acc(step=0, epoch=0, inference=True)
        else:
            self.board_vanilla_acc(step=0, epoch=0, inference=True)