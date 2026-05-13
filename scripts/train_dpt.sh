#!/bin/bash

# modify these augments if you want to try other datasets, splits or methods
# dataset: ['cityscapes', 'ade20k', 'coco']
# method: ['supervised_mask2former', 'supervised_dpt', 'sense_mask2former', 'sense_dpt']
# exp: just for specifying the 'save_path'
# split: Please check directory './splits/$dataset' for concrete splits
dataset='cityscapes'

# method='supervised_dpt'
# split='supervised'

method='sense_dpt'
split='generative'


config=configs/${dataset}_dpt.yaml

exp='dinov2_small' #  dinov2_base  dinov2_small

labeled_id_path=splits/$dataset/$split/labeled.txt
# unlabeled_id_path=splits/$dataset/$split/unlabeled_local.txt
unlabeled_id_path=splits/$dataset/$split/unlabeled.txt
save_path=exp/$dataset/$method/$exp/$split

nproc_per_node=1
master_port=12355
mkdir -p $save_path

python -m torch.distributed.launch \
    --nproc_per_node=$nproc_per_node \
    --master_addr=localhost \
    --master_port=$master_port \
    $method.py \
    --config=$config --labeled-id-path $labeled_id_path --unlabeled-id-path $unlabeled_id_path \
    --save-path $save_path --port $master_port 2>&1 | tee $save_path/out.log
