export CUDA_VISIBLE_DEVICES=0

python -u run.py \
  --task_name imputation \
  --is_training 1 \
  --root_path ./dataset/Humidity \
  --data_path Humidity_01234 \
  --model_id Humidity_01234 \
  --model DLinear \
  --data Humidity_Anomalous \
  --features M \
  --seq_len 100 \
  --pred_len 0 \
  --d_model 32 \
  --d_ff 64 \
  --e_layers 1 \
  --enc_in 1 \
  --anomaly_ratio 5 \
  --c_out 1 \
  --batch_size 16 \
  --train_epochs 10 \
  --learning_rate 0.001 \
  --mask_rate 0.2 \
