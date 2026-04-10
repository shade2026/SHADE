export CUDA_VISIBLE_DEVICES=0

python -u run.py \
  --task_name anomaly_detection \
  --is_training 1 \
  --root_path ./dataset/NO2 \
  --data_path NO2_01234 \
  --model_id NO2_01234 \
  --model SHADE \
  --data NO2_Anomalous \
  --features M \
  --seq_len 100 \
  --pred_len 0 \
  --d_model 32 \
  --d_ff 64 \
  --e_layers 1 \
  --top_k 2 \
  --loss_coef 0.1 \
  --loss_coef1 0.05 \
  --enc_in 1 \
  --c_out 1 \
  --anomaly_ratio 5 \
  --batch_size 16 \
  --train_epochs 1 \
  --learning_rate 0.001 \
  --mask_rate 0.2 \
  --interpolate 0 \
  --dmf_imputation_test 0 \
  --dl_imputation_test 0 \
  --imputation_model DLinear \
  --patch_size_list 20 10 5 2 10 5 4 2 20 10 4 2 \
  --num_nodes 32 \
  --residual_connection 1 \
  --revin 1 \
  --hyperedge_num 10 \
  --k_hyperedge 3 \
  --select_threshold MAD \

