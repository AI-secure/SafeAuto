#!/bin/bash
#####################
init_sigma=$1
final_sigma=$2
rag=$3         # true or false
topk=$4        # integer (only used if rag=true)

# Construct suffix based on rag and topk
if [ "$rag" = "true" ]; then
    suffix="rag_top${topk}"
    train_file="conversation_bddx_train_${suffix}.json"
    eval_file="conversation_bddx_eval_${suffix}.json"
else
    suffix="norag"
    train_file="conversation_bddx_train.json"
    eval_file="conversation_bddx_eval.json"
fi

formatted_init_sigma=$(printf "%.2f" $init_sigma)
formatted_final_sigma=$(printf "%.2f" $final_sigma)

MODEL_PATH="./checkpoints/Video-LLaVA-7B_safeauto_bddx_${formatted_init_sigma}-${formatted_final_sigma}_${suffix}"

DATA_ROOT="data"
HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=0 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 deepspeed llava/train/train_mem.py \
    --deepspeed ./scripts/zero3_offload.json \
    --output_dir ${MODEL_PATH} \
    --model_name_or_path ./checkpoints/Video-LLaVA-7B \
    --version v1 \
    --train_data_path data/conversation/bddx/${train_file} \
    --eval_data_path data/conversation/bddx/${eval_file} \
    --video_folder ${DATA_ROOT} \
    --image_folder ${DATA_ROOT} \
    --X "Video" "Image" \
    --video_tower ./cache_dir/LanguageBind_Video_merge \
    --image_tower ./cache_dir/LanguageBind_Image \
    --pretrain_mm_mlp_adapter checkpoints/Video-LLaVA-Pretrain-7B/mm_projector.bin \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_x_start_end False \
    --mm_use_x_patch_token False \
    --init_sigma $init_sigma \
    --final_sigma $final_sigma \
    --image_aspect_ratio pad \
    --group_by_modality_length False \
    --bf16 True \
    --num_train_epochs 2 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 300 \
    --save_total_limit 5 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to tensorboard \
    --cache_dir "./cache_dir"

echo "Finished training"