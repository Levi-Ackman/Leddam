#!/bin/bash


mkdir -p ./logs/LongForecasting/solar


export CUDA_VISIBLE_DEVICES=5
model_name=Leddam
pred_lens=( 336)
seq_lens=(96)
bss=(128)
lrs=(5e-4)

dropouts=(0.5)
log_dir="./logs/LongForecasting/solar/"


for pred_len in "${pred_lens[@]}"; do
  for seq_len in "${seq_lens[@]}"; do
    for bs in "${bss[@]}"; do
      for lr in "${lrs[@]}"; do
        for dropout in "${dropouts[@]}"; do
                  cmd="python -u run.py \
                    --task_name long_term_forecast \
                    --is_training 1 \
                    --root_path dataset/Solar/ \
                    --data_path solar_AL.txt \
                    --model_id "solar_${seq_len}_${pred_len}" \
                    --model $model_name \
                    --data Solar \
                    --features M \
                    --seq_len $seq_len \
                    --pred_len $pred_len \
                    --batch_size $bs \
                    --learning_rate $lr \
                    --enc_in 137 \
                    --dec_in 137 \
                    --c_out 137 \
                    --des 'Exp' \
                    --n_layers 3\
                    --d_model 512\
                    --pe_type no\
                    --dropout $dropout\
                    --itr 1 >${log_dir}${seq_len}_${pred_len}_${dropout}_bz${bs}_lr${lr}.log"

                  eval $cmd
        done
      done
    done
  done
done
