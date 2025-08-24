python train.py --dataset covid --lr=0.0001 --shortcut_type contrast --percent=0.01 \
    --valid_freq 1 --log_freq 2 --num_steps 3 \
    --method vanilla \
    --shorTest --local

#python train.py --dataset mimic --lr=0.0001 --shortcut_type mark --percent=0.05 \
#    --valid_freq 1 --log_freq 2 --num_steps 3 \
#    --method vanilla \
#    --shorTest --local
