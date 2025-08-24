import torch
from torchvision import models
import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import ssl_model.byol.builder as byol_builder
import ssl_model.byol.loader as loader
import ssl_model.simsiam.builder as simsiam_builder
import ssl_model.dino.vision_transformer as vision_transformer
import ssl_model.swav.src.resnet50 as swav_resnet
from pathlib import Path
vits = vision_transformer
import torch.nn as nn
import random
from util import torch_safe_save, get_device
def sample_unlabelled_images(bz,size):
    return torch.randn(bz, 1, size, size)

def sample_feature(bz,feature_dim):
    return torch.randn(bz,feature_dim)

def get_vanilla_bias_predictor(vanilla_ckpt_path,bias_ckpt_path):

    state_dict = torch.load(vanilla_ckpt_path)['state_dict']
    bias_state_dict = torch.load(bias_ckpt_path)['state_dict']
    del state_dict['fc.weight']
    del state_dict['fc.bias']
    state_dict['fc.weight'] = bias_state_dict['0.weight']
    state_dict['fc.bias'] = bias_state_dict['0.bias']
    num_classes = state_dict['fc.bias'].shape[0]
    predictor = models.__dict__['resnet18'](num_classes=num_classes)
    predictor.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    predictor.load_state_dict(state_dict)

    return predictor

def get_vanilla_predictor(ckpt_path):

    num_classes = torch.load(ckpt_path)['state_dict']['fc.bias'].shape[0]
    predictor = models.__dict__['resnet18'](num_classes=num_classes)
    predictor.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    predictor.load_state_dict(torch.load(ckpt_path)['state_dict'])

    return predictor

def get_linear_predictor(ckpt_path):

    num_classes,feature_dim = torch.load(ckpt_path)['state_dict']['0.weight'].shape
    predictor = torch.nn.Sequential(
        torch.nn.Linear(feature_dim, num_classes, bias=True),
    )
    predictor.load_state_dict(torch.load(ckpt_path)['state_dict'])

    return predictor

def get_byol_encoder(ckpt_path):

    projection_size = int(ckpt_path.rsplit('/',2)[-2].split('_dim_')[1].split('_')[0])
    projection_hidden_size = int(ckpt_path.rsplit('/',2)[-2].split('_hdim_')[1].split('_')[0])
    backbone = ckpt_path.rsplit('/',4)[-4]
    image_size = int(ckpt_path.rsplit('/',5)[-5])
    print(f'ssl encoder hidden dim {projection_hidden_size} feature dim {projection_size} backbone {backbone} image shape {image_size}')
    
    net = models.__dict__[backbone](weights=None)
    net.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)#modify torchvision
    model = byol_builder.BYOL(net, image_size=image_size, projection_hidden_size=projection_hidden_size, projection_size=projection_size)
    state_dict = torch.load(ckpt_path, map_location='cpu')['state_dict']

    encoder = model.online_encoder
    encoder.load_state_dict(state_dict)
    encoder = encoder.encoder_q
    encoder.fc = nn.Identity() 

    return encoder

def get_simsiam_encoder(ckpt_path=''):

    state_dict = torch.load(ckpt_path, map_location="cpu")['state_dict']
    projection_size = int(ckpt_path.rsplit('/',2)[-2].split('_dim_')[1].split('_')[0])
    prediction_hidden_size = int(ckpt_path.rsplit('/',2)[-2].split('_hdim_')[1].split('_')[0])
    arch = ckpt_path.rsplit('/')[-2].split('_')[0]
    dataset = ckpt_path.rsplit('/')[-4]

    backbone = models.__dict__[arch](num_classes=projection_size, zero_init_residual=True)
    if dataset in ['MIMIC','COVID']:
        backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)#modify torchvision
    encoder = simsiam_builder.SimSiam(backbone,projection_size, prediction_hidden_size).encoder

    for k in list(state_dict.keys()):
        # retain only encoder up to before the embedding layer
        if k.startswith('encoder'):
            # remove prefix
            state_dict[k[len("encoder."):]] = state_dict[k]
        # delete renamed or unused k
        del state_dict[k]

    encoder.load_state_dict(state_dict, strict=False)

    return encoder

def get_swav_encoder(ckpt_path=''):
    arch = ckpt_path.split('/')[-3].split('_')[0]
    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)['state_dict']
    print(list(state_dict))
    backbone = swav_resnet.__dict__[arch](output_dim=0, eval_mode=True)
    backbone.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(2, 2), bias=False)
    #print(list(backbone.state_dict().keys()))
    msg = backbone.load_state_dict(state_dict, strict=False)
    print("Load pretrained model with msg: {}".format(msg))
    class MyEncoder(torch.nn.Module):
        def __init__(self,backbone):
            super(MyEncoder, self).__init__()
            self.backbone = backbone
            self.backbone.eval()
            self.av_pool = nn.AdaptiveAvgPool2d((1, 1))

        def forward(self, x):
            x = self.backbone(x)
            x = self.av_pool(x)
            x = x.view(x.size(0), -1)
            return x
    encoder = MyEncoder(backbone=backbone)
    for key in list(encoder.backbone.state_dict()):
        assert (encoder.backbone.state_dict()[key]==backbone.state_dict()[key]).all()

    return encoder

def get_dino_encoder(ckpt_path=''):
    checkpoint_key = 'teacher'# or 'student'
    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    arch = ckpt_path.split('/')[-2].split('_')[0]
    dataset = ckpt_path.split('/')[-3].replace('_result','')
    size = 256 if dataset == 'mimic' else 299
    if arch in vits.__dict__.keys():
        encoder = vits.__dict__[arch](img_size = [size],in_chans = 1,patch_size=16, num_classes=0)

    # if the network is a XCiT
    elif "xcit" in arch:
        encoder = torch.hub.load('facebookresearch/xcit:main', arch, num_classes=0)

    # otherwise, we check if the architecture is in torchvision models
    elif arch in models.__dict__.keys():
        encoder = models.__dict__[arch]()
        encoder.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)#modify torchvision
        encoder.fc = nn.Identity() 
    else:
        print(f"Unknow architecture: {arch}")
        raise NotImplementedError
    
    if checkpoint_key is not None and checkpoint_key in state_dict:
        print(f"Take key {checkpoint_key} in provided checkpoint dict")
        state_dict = state_dict[checkpoint_key]
    # remove `module.` prefix
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    # remove `backbone.` prefix induced by multicrop wrapper
    state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
    msg = encoder.load_state_dict(state_dict, strict=False)
    print('Pretrained weights found at {} and loaded with msg: {}'.format(ckpt_path, msg))

    return encoder

def get_simclr_encoder(ckpt_path):
    state_dict = torch.load(ckpt_path, map_location="cpu")
    arch = ckpt_path.split('/')[-2].split('_')[0]
    state_dict = state_dict['state_dict']

    encoder = models.__dict__[arch](weights=None)
    encoder.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    encoder.fc = nn.Identity() 
    for k in list(state_dict.keys()):

        if k.startswith('backbone.'):
          if k.startswith('backbone') and not k.startswith('backbone.fc'):
            # remove prefix
            state_dict[k[len("backbone."):]] = state_dict[k]
        del state_dict[k]
    encoder.load_state_dict(state_dict, strict=False)

    return encoder

def get_ssl_info(ckpt_path):
    aug = ckpt_path.split('aug_')[1].split('_epochs')[0].lower()
    assert aug in ['cjb','jb','cb','cj']
    remove_str_lst = ['ckp-','checkpoint','_','.','pth','tar']
    ckpt_name = ckpt_path.rsplit("/",2)[-1]
    for str in remove_str_lst:
        ckpt_name = ckpt_name.replace(str,'')
    return aug, int(ckpt_name)

def get_encoder(ssl_type,ssl_ckpt_path):
    if ssl_type=='byol':
        encoder = get_byol_encoder(ssl_ckpt_path)
    elif ssl_type=='simsiam':
        encoder = get_simsiam_encoder(ssl_ckpt_path)
    elif ssl_type=='dino':
        encoder = get_dino_encoder(ssl_ckpt_path)
    elif ssl_type=='swav':
        encoder = get_swav_encoder(ssl_ckpt_path)
    elif ssl_type=='simclr':
        encoder = get_simclr_encoder(ssl_ckpt_path)
    else:
        raise NotImplementedError
    
    encoder = encoder.to(get_device())
    return encoder

def gen_flip_path_list(path_list,flip_size=3):
    #path_list = path_list[-500:-200]
    flip_path_list = list()
    for _ in range(flip_size):
        flip_path_list.append(list())
    for path in path_list:

        shortcut_label = int(path.rsplit('.',1)[0][-1])
        flip_str_head = path.replace('test','test_flip').replace('val','val_flip').rsplit('_',1)[0]+'_'
        flip_str_tail = path.rsplit('.',1)[-1]#png,jpg
        tmp = 0
        for flip_label in range(flip_size+1):
            if flip_label != shortcut_label:
                flip_path = f"{flip_str_head}{flip_label}.{flip_str_tail}"
                flip_path_list[tmp].append(flip_path)
                tmp += 1

    for i in range(flip_size):
        assert len(path_list) == len(flip_path_list[i])
        for j in range(len(flip_path_list[i])):
            assert flip_path_list[i][j].rsplit('_',1)[0] == path_list[j].replace('test','test_flip').replace('val','val_flip').rsplit('_',1)[0]
    return flip_path_list

def batch_gen_feature(encoder, img_path_list, transforms, bz=32):

    features_list = []
    batch_imgs = []

    for index in tqdm(range(len(img_path_list))):
        img = Image.open(img_path_list[index])
        img_tensor = transforms(img).unsqueeze(0).to(get_device())
        batch_imgs.append(img_tensor)

        if len(batch_imgs) == bz or index == len(img_path_list) - 1:
            cat_img = torch.cat(batch_imgs, dim=0)
            with torch.no_grad():
                feature = encoder(cat_img).cpu().numpy()
            features_list.append(feature)
            batch_imgs = []

    cat_feature = np.concatenate(features_list, axis=0)
    return cat_feature

def generate_feature(dataset,img_path_list,ssl_ckpt_path,ssl_type,shortcut_type,shortcut_skew,transforms,split,use_bias_label):
    _, epoch = get_ssl_info(ssl_ckpt_path)
    ssl_f_name = ssl_ckpt_path.rsplit("/",2)[-2]

    ssl_shortcut_info = ssl_ckpt_path.rsplit("/",3)[-3]
    feature_dir = f'feature/ssl_feature/{dataset}/{ssl_type}/{shortcut_type}{shortcut_skew}/{ssl_shortcut_info}_{ssl_f_name}/ep{epoch}'

    if not os.path.exists(feature_dir):
        os.makedirs(feature_dir)
    feature_fname = os.path.join(feature_dir,f'{split}_ssl_features.npy')
    flip_feature_fname = feature_fname.replace('_ssl','_flip_ssl')
    if os.path.exists(feature_fname) :
        print(feature_fname)
        return feature_fname, flip_feature_fname

    encoder = get_encoder(ssl_type,ssl_ckpt_path).eval()
    
    tmp_fname = Path("tmp") / f'tmp{random.randint(0,100000)}.pth.tar'
    torch_safe_save(encoder.state_dict(), tmp_fname)
    
    flip_size = 3

    feature = batch_gen_feature(encoder,img_path_list,transforms)
    np.save(feature_fname,feature)

    if split not in ['train','val_flip','test_flip'] and shortcut_type not in ['no','LO','Male','Female','Age','Race']:
        
        flip_path_list = gen_flip_path_list(img_path_list,flip_size=flip_size)
        flip_feature = np.zeros((0,feature.shape[0],feature.shape[1]))
        for i in range(flip_size):
            flip_feature_ = batch_gen_feature(encoder,flip_path_list[i],transforms)
            flip_feature = np.concatenate((flip_feature,np.expand_dims(flip_feature_,axis=0)))
        np.save(flip_feature_fname,flip_feature)
    else:
        flip_feature_fname = None
    state_dict = torch.load(tmp_fname)
    for k in list(state_dict.keys()):
        assert ((state_dict[k].cpu() == encoder.state_dict()[k].cpu()).all())
    os.remove(tmp_fname)
    return feature_fname, flip_feature_fname