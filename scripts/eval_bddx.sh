#!/bin/bash
#####################
init_sigma=$1
final_sigma=$2
rag=$3         # true or false
topk=$4        # integer (only used if rag=true)
cuda_id=${5:-0} # CUDA device to use for evaluation

# Construct suffix and filenames
if [ "$rag" = "true" ]; then
    suffix="rag_top${topk}"
    eval_file="conversation_bddx_eval_${suffix}.json"
else
    suffix="norag"
    eval_file="conversation_bddx_eval.json"
fi

# Format sigmas
formatted_init_sigma=$(printf "%.2f" $init_sigma)
formatted_final_sigma=$(printf "%.2f" $final_sigma)

# Model checkpoint path
MODEL_PATH="./checkpoints/Video-LLaVA-7B_safeauto_bddx_${formatted_init_sigma}-${formatted_final_sigma}_${suffix}"

# Run evaluation
CUDA_VISIBLE_DEVICES=$cuda_id python -m llava.serve.eval_custom_predsig_bddx \
    --model-path ${MODEL_PATH} \
    --input data/conversation/bddx/${eval_file} \
    --output "results/bddx_${formatted_init_sigma}-${formatted_final_sigma}_${suffix}"

echo "Finished inference. Output saved to: results/bddx_${formatted_init_sigma}-${formatted_final_sigma}_${suffix}"