#!/bin/bash
model_name=CNN

# Define two arrays
type_filter=("none" "BPF_20_200" "none" "BPF_20_200" "none" "BPF_20_200" "none" "BPF_20_200")
type_norm=("none" "none" "mvc" "mvc" "min_max" "min_max" "standardization" "standardization")

for (( i=0; i<${#type_filter[@]}; i++ )); do
    
    # echo ${type_filter[$i]} ${type_norm[$i]}

    current_time=$(date +%Y%m%d_%H%M)
    log_name="${model_name}_${current_time}"
    mkdir -p Results/$log_name

    CUDA_VISIBLE_DEVICES=1 \
    python3 main_develop_DB2.py \
    --exercise_list 1,2,3 \
    --window_size_sec 0.2  \
    --window_step_sec 0.1 \
    --num_epoch 3000 \
    --batch_size 1024 \
    --model_type $model_name \
    --lr 0.001 \
    --en_train \
    --type_filter ${type_filter[$i]} \
    --type_norm  ${type_norm[$i]} \
    --log_name $log_name \
    | tee Results/$log_name/record.log

    # --en_train \
    # --feat_extract \
    # --subject_list 1 \
    # --load_dataset \

done