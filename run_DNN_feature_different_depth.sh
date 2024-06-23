#!/bin/bash
model_name=DNN_feature

dropout_DNN=(0.5)

for (( i=0; i<${#dropout_DNN[@]}; i++ )); do
    
    # echo ${type_filter[$i]} ${type_norm[$i]}

    current_time=$(date +%Y%m%d_%H%M)
    log_name="${model_name}_${current_time}_Dropout_${dropout_DNN[$i]}"
    mkdir -p Results/$log_name

    CUDA_VISIBLE_DEVICES=1 \
    python3 main_inter_subject_p2.py \
    --subject_list=1,2,3,4,5 \
    --exercise_list 1,2,3 \
    --window_size_sec 0.2  \
    --window_step_sec 0.01 \
    --num_epoch 500 \
    --batch_size 1024 \
    --model_type $model_name \
    --lr 0.001 \
    --dropout_DNN ${dropout_DNN[$i]} \
    --en_train \
    --feat_extract \
    --type_filter none \
    --type_norm  standardization \
    --log_name $log_name \
    | tee Results/$log_name/record.log

    # --en_train \
    # --feat_extract \
    # --subject_list 1 \
    # --load_dataset \
done

