import argparse

def get_parser():
    parser = argparse.ArgumentParser()

    # training
    parser.add_argument("--batch_size", help="batch_size", default=256, type=int)
    parser.add_argument("--lr",help='learning rate',default=1e-4, type=float)
    parser.add_argument("--weight_decay",help='weight_decay',default=0.0, type=float)
    parser.add_argument("--momentum",help='momentum',default=0.9, type=float)
    parser.add_argument("--num_workers", help="workers number", default=1, type=int)
    parser.add_argument("--exp", help='experiment name', default='', type=str)
    parser.add_argument("--device", help="cuda or cpu", default=None, type=str)
    
    parser.add_argument("--dataset", help="data to train",choices=['mimic','mimic_ssl','covid','covid_ssl'], default= 'mimic', type=str)

    parser.add_argument("--percent", help="percentage of conflict", default= "0.0", type=str)
    parser.add_argument("--shortcut_type", help='mimic shortcut', default='no', type=str)
    parser.add_argument("--group_type", help='mimic demographic infomation for fairness',choices=['age','gender','race','no','LO'], default='no', type=str)

    parser.add_argument('--continue_train','-c', action='store_true', default=False,help='continue train.')
    parser.add_argument('--local','-l', action='store_true',help='not use wandb, tensorboard')
    # logging
    parser.add_argument("--result_root", help='path for saving model', default='./result', type=str)

    # experiment
    parser.add_argument("--model", default='',type=str, help="model name")
    parser.add_argument("--method", default='vanilla',choices=['aug_vanilla','vanilla','fit'],type=str, help="train vanilla cnn or fit other")
    parser.add_argument("--shorTest", action="store_true", help="whether to train vanilla with shorcutTest")
    parser.add_argument("--ssl_ckpt_path", default='',type=str, help="ssl encoder path")
    parser.add_argument("--ssl_type", default='',type=str,choices=['','simsiam','byol','simclr','dino','swav'], help="ssl encoder path")
    parser.add_argument('--use_bias_label', action='store_true',help='use for training on bias label')
    return parser