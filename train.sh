#!/bin/bash

tmux new -s gaussian1 'python train_gaussian_map.py; tmux kill-session -t mysession'

### KITTI WEAKLY SUPERVISED

python "./train_KITTI_weak_nips_orienternet.py" \
  --rotation_range 0 \
  --stage 4 \
  --share 1 \
  --level 1 \
  --ConfGrd 1 \
  --contrastive_coe 1 \
  --name "orienternet_weakly_GPS" \
  --batch_size 8 \
  --epochs 10 \
  --test 0 \
  --visualize 0 

python "./train_KITTI_weak_nips_vfa.py" \
  --rotation_range 0 \
  --stage 4 \
  --share 1 \
  --level 1 \
  --ConfGrd 1 \
  --contrastive_coe 1 \
  --name "vfa_weakly_GPS" \
  --batch_size 8 \
  --epochs 10 \
  --test 0 \
  --visualize 0 


python "./train_KITTI_weak_weather.py" \
  --rotation_range 0 \
  --stage 4 \
  --share 1 \
  --level 1 \
  --ConfGrd 1 \
  --contrastive_coe 1 \
  --name "feat32_offset_0.5_confidence_original" \
  --batch_size 8 \
  --epochs 10 \
  --test 1 \
  --visualize 0 

#6.25e-5 cos GPS
python "./train_KITTI_weak_nips.py" \
  --rotation_range 10 \
  --stage 1 \
  --share 1 \
  --level 1 \
  --ConfGrd 1 \
  --GPS_error_coe 0 \
  --name "feat32_depth" \
  --batch_size 8 \
  --epochs 10 \
  --test 0 \
  --visualize 0
    
python "./train_KITTI_weak_seq.py" \
  --rotation_range 0 \
  --stage 4 \
  --share 1 \
  --level 1 \
  --ConfGrd 1 \
  --contrastive_coe 1 \
  --name "feat32_offset_0.5_seq3_6.5e-5" \
  --batch_size 8 \
  --epochs 10 \
  --test 0 \
  --visualize 0 \
  --sequence 3


### VIGOR WEAKLY SUPERVISED

### same area test
python "./train_vigor_2DoF.py" \
  --rotation_range 0 \
  --share 0 \
  --ConfGrd 1 \
  --level 1 \
  --Supervision "Weakly" \
  --area "same" \
  --name 'vigor_0.3_3.0_70_1.25e-4_depth' \
  --batch_size 32 \
  --test 1 \
  --lr 1.25e-4


### same area train

python "./train_vigor_2DoF.py" \
  --rotation_range 0 \
  --share 0 \
  --ConfGrd 1 \
  --level 1 \
  --Supervision "Weakly" \
  --area "same" \
  --name 'vigor_0.3_3.0_70_1.25e-4_depth' \
  --batch_size 8 \
  --test 0 \
  --lr 1.25e-4 \
  --epoch 15

### cross area test
python "./train_vigor_2DoF.py" \
  --rotation_range 0 \
  --share 0 \
  --ConfGrd 1 \
  --level 1 \
  --Supervision "Weakly" \
  --area "cross" \
  --name 'vigor_0.3_3.0_70_1.25e-4_depth' \
  --batch_size 8 \
  --test 1 \
  --lr 1.25e-4 \
  --epoch 15


### cross area train
python "./train_vigor_2DoF.py" \
  --rotation_range 0 \
  --share 0 \
  --ConfGrd 1 \
  --level 1 \
  --Supervision "Weakly" \
  --area "cross" \
  --name 'vigor_0.3_3.0_60_1e-4_depth' \
  --batch_size 8 \
  --test 0 \
  --lr 1e-4 \
  --epoch 15

### same area train with GPS noise
python "./train_vigor_2DoF.py" \
  --rotation_range 0 \
  --share 0 \
  --ConfGrd 1 \
  --level 1 \
  --Supervision "Weakly" \
  --area "same" \
  --name 'vigor_0.3_3.0_70_1.25e-4_depth' \
  --batch_size 8 \
  --test 0 \
  --GPS_error_coe 1 \
  --lr 2e-4 \
  --epoch 15


### cross area train with GPS noise
python "./train_vigor_2DoF.py" \
  --rotation_range 0 \
  --share 0 \
  --ConfGrd 1 \
  --level 1 \
  --Supervision "Weakly" \
  --area "cross" \
  --name 'vigor_0.3_3.0_70_1e-4_depth' \
  --batch_size 8 \
  --test 0 \
  --GPS_error_coe 1 \
  --lr 1e-4 \
  --epoch 15