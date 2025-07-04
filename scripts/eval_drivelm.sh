#!/bin/bash
#####################
init_sigma=$1
final_sigma=$2
rag=$3
topk=$4
cuda_id=${5:-0}

if [ "$rag" = "true" ]; then
    suffix="rag_top${topk}"
    eval_file="conversation_drivelm_eval_${suffix}.json"
else
    suffix="norag"
    eval_file="conversation_drivelm_eval.json"
fi

formatted_init_sigma=$(printf "%.2f" $init_sigma)
formatted_final_sigma=$(printf "%.2f" $final_sigma)

MODEL_PATH="./checkpoints/Video-LLaVA-7B_safeauto_drivelm_${formatted_init_sigma}-${formatted_final_sigma}_${suffix}"

CUDA_VISIBLE_DEVICES=$cuda_id python -m llava.serve.eval_custom_predsig_drivelm \
    --model-path ${MODEL_PATH} \
    --input data/conversation/bddx/${eval_file} \
    --output "results/drivelm_${formatted_init_sigma}-${formatted_final_sigma}_${suffix}"

echo "Finished inference. Output saved to results/drivelm_${formatted_init_sigma}-${formatted_final_sigma}_${suffix}"
