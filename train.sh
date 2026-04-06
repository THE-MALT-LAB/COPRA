export DEBUG_MODE="true"
export CUDA_VISIBLE_DEVICES=7
export HF_HOME="/aiau001_scratch/dzj0055/HYWUVLM-R1/cache/"
export PYTHONPATH="HY-WU/:VLM-R1/src/open-r1-multimodal/:$PYTHONPATH"
RUN_NAME="InternVL2-8B-GRPO-VAD"
export LOG_PATH="./debug_log_$RUN_NAME.txt"

torchrun --nproc_per_node="1" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12346" \
    grpo.py \
    --output_dir /aiau001_scratch/dzj0055/HYWUVLM-R1/my_output/$RUN_NAME \
    --model_name_or_path OpenGVLab/InternVL2-8B \
    --dataset_name annotations/vad.yaml \
    --image_root /aiau001_scratch/dzj0055/ucf_crime/videos/ \
    --freeze_vision_modules true \
    --max_anyres_num 1 \
    --num_generations 4 \
    --per_device_train_batch_size 4 \
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
    --beta 0.0