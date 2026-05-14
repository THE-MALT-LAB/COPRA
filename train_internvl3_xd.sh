export DEBUG_MODE="true"
export CUDA_VISIBLE_DEVICES=7
export HF_HOME="/aiau001_scratch/dzj0055/HYWUVLM-R1/cache/"
export TRITON_CACHE_DIR="/aiau001_scratch/dzj0055/HYWUVLM-R1/cache/triton/"
export PYTHONPATH="HY-WU/:VLM-R1/src/open-r1-multimodal/:$PYTHONPATH"
RUN_NAME="InternVL3-8B-GRPO-VAD-XDViolence"
export LOG_PATH="./debug_log_$RUN_NAME.txt"

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True torchrun --nproc_per_node="1" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12346" \
    grpo.py \
    --output_dir /aiau001_scratch/dzj0055/HYWUVLM-R1/my_output/$RUN_NAME \
    --model_name_or_path OpenGVLab/InternVL3-8B \
    --dataset_name annotations/vad_xd.yaml \
    --image_root /aiau001_scratch/dzj0055/xd_violence/videos/ \
    --freeze_vision_modules true \
    --max_anyres_num 1 \
    --num_iterations 1 \
    --num_generations 3 \
    --per_device_train_batch_size 3 \
    --gradient_accumulation_steps 1 \
    --logging_steps 1 \
    --bf16 \
    --torch_dtype bfloat16 \
    --data_seed 42 \
    --report_to wandb \
    --gradient_checkpointing false \
    --attn_implementation flash_attention_2 \
    --num_train_epochs 1 \
    --run_name $RUN_NAME \
    --save_steps 400 \
    --save_only_model true \
    --beta 0.1 \
    --max_grad_norm 1