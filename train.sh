#!/bin/bash
export WANDB_PROJECT=Video-GRPO
export WANDB_NAME=Qwen2.5_7b_TG

export TRANSFOMERS_CACHE="/aiau001_scratch/dzj0055/HYWUChat-R1.5/cache/"
export PYTHONPATH="VideoChat-R1/Videochat-R1.5/:HY-WU/:$PYTHONPATH"
OUTDIR=ckpt/cotraining

export DEBUG_MODE="true"
export LOG_PATH="./log.txt"
export CUDA_VISIBLE_DEVICES="3"

torchrun --nnodes=1 \
    --nproc-per-node=1 \
    grpo.py \
    --deepspeed /home/cse_h2/dzj0055/HYWUChat-R1.5/VideoChat-R1/Videochat-R1.5/scripts/zero3_offload.json \
    --model_name_or_path "Qwen/Qwen2-VL-7B-Instruct" \
    --output_dir $OUTDIR \
    --training_data /home/cse_h2/dzj0055/HYWUChat-R1.5/annotations/dataset_training_file.json \
    --dataset_name ucf_crime \
    --max_completion_length 1024 \
    --num_generations 2 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --logging_steps 1 \
    --bf16 \
    --torch_dtype bfloat16 \
    --data_seed 42 \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --num_train_epochs 1 \
    --run_name $WANDB_NAME \
    --report_to tensorboard \
    --save_steps 1000 \
    --learning_rate 1e-6 \
    --save_only_model true
