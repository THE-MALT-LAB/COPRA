#!/bin/bash
export WANDB_PROJECT=Video-GRPO
export WANDB_NAME=Qwen2.5_7b_TG

export TRANSFOMERS_CACHE="/aiau001_scratch/dzj0055/HYWUChat-R1.5/cache/"
export PYTHONPATH="VideoChat-R1/Videochat-R1.5/:HY-WU/:$PYTHONPATH"
OUTDIR=ckpt/cotraining

export DEBUG_MODE="true"
export LOG_PATH="./log.txt"
export CUDA_VISIBLE_DEVICES="0,1,2,3,6"

# Removed --deepspeed /home/cse_h2/dzj0055/HYWUChat-R1.5/VideoChat-R1/Videochat-R1.5/scripts/zero3_offload.json \
# This shards optimizer parameters, gradients, and model parameters.
# we could try sharding just optimizer states using zero-1. not necessary atm.
accelerate launch --num_processes 5 eval/eval_vid_level_pg_xd.py