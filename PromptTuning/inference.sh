python3 train.py \
 --data_path '../../../../Data/Data/XsumDetect/' \
 --training_mode 'full-shot' \
 --log_path './log/' \
 --model_path '../../../Data/PLM/BARTCNN' \
 --saved_model_path "../../../../Data/PLM/PromptBART/" \
 --epoch 500 \
 --lr 0.001 \
 --eps 1e-8 \
 --task_name 'training' \
 --predict_ratio 0.55

