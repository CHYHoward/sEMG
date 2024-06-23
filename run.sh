# timestamp=$(date +%Y%m%d_%H%M)
CUDA_VISIBLE_DEVICES=1
python3 main_develop_DB2.py \
--exercise_list 1,2,3 \
--window_size_sec 0.2  \
--window_step_sec 0.1 \
--num_epoch 2000 \
--batch_size 512 \
--model_type ViT \
--lr 0.001 \
--en_train \
| tee log/$(date +%Y%m%d_%H%M)_main_develop_DB2.log 

# --en_train \
# --feat_extract \
# --subject_list 1 \
# --load_dataset \