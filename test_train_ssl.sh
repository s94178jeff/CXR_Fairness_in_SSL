#python train.py --dataset covid_ssl --lr=0.001 --shortcut_type contrast --percent=0.05 --method vanilla \
#    --valid_freq 2 --log_freq 2 --num_steps 5 \
#    --ssl_type 'byol' \
#    --ssl_ckpt_path '/Volumes/One Touch/LWBC/SSL/covid_checkpoints/byol_finetune/299/resnet18/contrast0.05/resnet18_bs_128_dim_2048_hdim_512_aug_CJB_epochs_150/checkpoint_0149.pth.tar' \
#    --use_bias_label \
#    --local

#python train.py --dataset covid_ssl --lr=0.001 --shortcut_type contrast --percent=0.05 --method vanilla \
#    --valid_freq 2 --log_freq 2 --num_steps 5 \
#    --ssl_type 'simclr' \
#    --ssl_ckpt_path '/Volumes/One Touch/SimCLR/covid_result/contrast0.05/resnet18_bz256_dim128_aug_CJB_epochs180/checkpoint_180.pth.tar' \
#    --use_bias_label \
#    --local

#python train.py --dataset covid_ssl --lr=0.001 --shortcut_type mark --percent=0.01 --method vanilla \
#    --valid_freq 2 --log_freq 2 --num_steps 5\
#    --ssl_type 'swav' \
#    --ssl_ckpt_path '/Volumes/One Touch/swav/covid_result/mark0.01/resnet18_bs64_aug_cjb_epochs_180/checkpoints/ckp-179.pth' \
#    --use_bias_label \
#    --local

#python train.py --dataset covid_ssl --lr=0.001 --shortcut_type no --percent=0.0 --method vanilla \
#    --valid_freq 2 --log_freq 2 --num_steps 5\
#    --ssl_type 'dino' \
#    --ssl_ckpt_path '/Volumes/One Touch/dino/mimic_result/no0.0/resnet50_bz64_dim60000_aug_cjb_epochs_180/checkpoint0179.pth' \
#    --local


python train.py --dataset covid_ssl --lr=0.001 --shortcut_type no --percent=0.0 --method vanilla \
    --valid_freq 2 --log_freq 2 --num_steps 5\
    --local