#!/bin/bash
export WANDB_PROJECT=Video-GRPO
export WANDB_NAME=Qwen2.5_7b_TG

export HF_HOME="/home/darryl/HYWUVLM-R1/env/cache/"
export PYTHONPATH="HY-WU/:VLM-R1/src/open-r1-multimodal/:$PYTHONPATH"
OUTDIR=ckpt/cotraining

export DEBUG_MODE="true"
export LOG_PATH="./log.txt"
export CUDA_VISIBLE_DEVICES="0,1,2,3"

# Removed --deepspeed /home/cse_h2/dzj0055/HYWUChat-R1.5/VideoChat-R1/Videochat-R1.5/scripts/zero3_offload.json \
# This shards optimizer parameters, gradients, and model parameters.
# we could try sharding just optimizer states using zero-1. not necessary atm.
accelerate launch --num_processes 4 eval/eval_vru_accident_densecap.py