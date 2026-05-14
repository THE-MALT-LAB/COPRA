#!/bin/bash
export WANDB_PROJECT=Video-GRPO
export WANDB_NAME=Qwen2.5_7b_TG

export HF_HOME="/aiau001_scratch/dzj0055/HYWUVLM-R1/cache/"
export PYTHONPATH="HY-WU/:VLM-R1/src/open-r1-multimodal/:$PYTHONPATH"
OUTDIR=ckpt/cotraining

export DEBUG_MODE="true"
export LOG_PATH="./log.txt"
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
#export CUDA_VISIBLE_DEVICES="7"

# Results in Holmes-VAU paper may be questionable
# https://github.com/pipixin321/HolmesVAU/issues/17
accelerate launch --num_processes 8 eval/eval_vid_level_pg_hivau_70k.py