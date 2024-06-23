# timestamp=$(date +%Y%m%d_%H%M)
model_name=ViT_TNet
current_time=$(date +%Y%m%d_%H%M)
log_name="${model_name}_${current_time}"
mkdir -p Results/$log_name

    CUDA_VISIBLE_DEVICES=1 \
    python3 main_intra_subject.py \
    --subject_list 1,2,3,4,5,6,7,8,9,10 \
    --exercise_list 1,2,3 \
    --window_size_sec 0.2  \
    --window_step_sec 0.01 \
    --num_epoch 1000 \
    --batch_size 1024 \
    --model_type $model_name \
    --lr 0.0001 \
    --en_train \
    --type_filter none \
    --type_norm standardization \
    --log_name $log_name \
    | tee Results/$log_name/record.log

# --en_train \
# --feat_extract \
# --load_dataset \
# --pretrain_model_PATH Results/ViT_FNet_20231114_0144/ViT_FNet.pth \f
# --pretrain_model_PATH Results/ViT_TNet_20231114_1138_Pretrain_on_all_subject/ViT_TNet.pth \
#--pretrain_model_PATH Results/ViT_TNet_20231114_1138_Pretrain_on_all_subject/ViT_TNet.pth \