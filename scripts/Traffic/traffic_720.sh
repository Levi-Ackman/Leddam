#!/bin/bash


mkdir -p ./logs/LongForecasting/traffic


export CUDA_VISIBLE_DEVICES=6
model_name=Leddam
pred_lens=(720)
seq_lens=(96)
bss=(32)
lrs=(1e-3)
dropouts=(0.5)

log_dir="./logs/LongForecasting/traffic/"


for pred_len in "${pred_lens[@]}"; do
  for seq_len in "${seq_lens[@]}"; do
    for bs in "${bss[@]}"; do
      for lr in "${lrs[@]}"; do
        for dropout in "${dropouts[@]}"; do
                  cmd="python -u run.py \
                    --task_name long_term_forecast \
                    --is_training 1 \
                    --root_path dataset/traffic/ \
                    --data_path traffic.csv \
                    --model_id "traffic_${seq_len}_${pred_len}" \
                    --model $model_name \
                    --data custom \
                    --features M \
                    --seq_len $seq_len \
                    --pred_len $pred_len \
                    --batch_size $bs \
                    --learning_rate $lr \
                    --enc_in 862 \
                    --dec_in 862 \
                    --c_out 862 \
                    --des 'Exp' \
                    --n_layers 3\
                    --d_model 256\
                    --pe_type no\
                    --dropout $dropout\
                    --itr 1 >${log_dir}${seq_len}_${pred_len}_${dropout}_bz${bs}_lr${lr}.log"

                  eval $cmd
        done
      done
    done
  done
done
