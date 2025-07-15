#!/bin/bash

### NEED TO CHANGE ###
save_checkpoint_path="checkpoints/q-seer-libero"
root_dir="/data1/shujunyang/libero"
vit_checkpoint_path="checkpoints/vit_mae/mae_pretrain_vit_base.pth"
finetune_from_pretrained_ckpt="checkpoints/33.pth"
libero_path="/data2/shujunyang/Seer-libero/LIBEROO"
### NEED TO CHANGE ###
calvin_dataset_path="calvin/dataset/task_ABC_D"

node=1
node_num=8
# export CUDA_VISIBLE_DEVICES='0,1,2,3'
torchrun --nnodes=${node} --nproc_per_node=${node_num} --master_port=10211 q_train.py \
    --traj_cons \
    --rgb_pad 10 \
    --gripper_pad 4 \
    --gradient_accumulation_steps 4 \
    --bf16_module "vision_encoder" \
    --vit_checkpoint_path ${vit_checkpoint_path} \
    --calvin_dataset ${calvin_dataset_path} \
    --workers 8 \
    --lr_scheduler cosine \
    --save_every_iter 100000 \
    --num_epochs 20 \
    --seed 42 \
    --batch_size 16 \
    --precision fp32 \
    --learning_rate 1e-4 \
    --save_checkpoint \
    --finetune_type libero_finetune \
    --root_dir ${root_dir} \
    --wandb_project seer-libero-q_model \
    --weight_decay 1e-4 \
    --num_resampler_query 6 \
    --run_name finetune_Seer-libero_to_q-model \
    --save_checkpoint_path ${save_checkpoint_path} \
    --transformer_layers 24 \
    --phase "finetune" \
    --obs_pred \
    --value_pred_steps 1 \
    --sequence_length 7 \
    --future_steps 3 \
    --window_size 10 \
    --loss_image \
    --loss_value \
    --loss_action \
    --reset_action_token \
    --reset_obs_token \
    --save_checkpoint_seq 1 \
    --start_save_checkpoint -1 \
    --gripper_width \
    --warmup_epochs 1 \
    --libero_path ${libero_path} \
    --finetune_from_pretrained_ckpt ${finetune_from_pretrained_ckpt} \
    --report_to_wandb \



