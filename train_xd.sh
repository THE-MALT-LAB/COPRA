#!/bin/bash
export WANDB_PROJECT=Video-GRPO
export WANDB_NAME=Qwen2.5_7b_TG_XDViolence

export TRANSFORMERS_CACHE="/aiau001_scratch/dzj0055/HYWUChat-R1.5/cache/"
export PYTHONPATH="VideoChat-R1/Videochat-R1.5/:HY-WU/:$PYTHONPATH"
OUTDIR=ckpt/cotraining_xd

export DEBUG_MODE="true"
export LOG_PATH="./log.txt"
export CUDA_VISIBLE_DEVICES="7"

# Removed --deepspeed /home/cse_h2/dzj0055/HYWUChat-R1.5/VideoChat-R1/Videochat-R1.5/scripts/zero3_offload.json \
# This shards optimizer parameters, gradients, and model parameters.
# we could try sharding just optimizer states using zero-1. not necessary atm.

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True torchrun --nnodes=1 \
    --nproc-per-node=1 \
    grpo.py \
    --model_name_or_path "Qwen/Qwen2-VL-7B-Instruct" \
    --output_dir $OUTDIR \
    --training_data /home/cse_h2/dzj0055/HYWUChat-R1.5/annotations/dataset_training_file_xd.json \
    --dataset_name xd_violence \
    --max_completion_length 256 \
    --num_generations 4 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --logging_steps 1 \
    --bf16 \
    --torch_dtype bfloat16 \
    --data_seed 42 \
    --gradient_checkpointing false \
    --attn_implementation flash_attention_2 \
    --num_train_epochs 1 \
    --run_name $WANDB_NAME \
    --report_to wandb \
    --save_steps 1000 \
    --learning_rate 1e-6 \
    --save_only_model true \
    --beta 0.1