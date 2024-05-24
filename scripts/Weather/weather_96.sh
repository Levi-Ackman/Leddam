#!/bin/bash


mkdir -p ./logs/LongForecasting/weather


export CUDA_VISIBLE_DEVICES=7
model_name=Leddam
pred_lens=( 96)
seq_lens=(96)
bss=(128)
lrs=(5e-4)
dropouts=(0.0)
d_models=(256)

log_dir="./logs/LongForecasting/weather/"


for pred_len in "${pred_lens[@]}"; do
  for seq_len in "${seq_lens[@]}"; do
    for bs in "${bss[@]}"; do
      for lr in "${lrs[@]}"; do
        for dropout in "${dropouts[@]}"; do
          for d_model in "${d_models[@]}"; do
                  cmd="python -u run.py \
                    --task_name long_term_forecast \
                    --is_training 1 \
                    --root_path dataset/weather/ \
                    --data_path weather.csv \
                    --model_id "weather_${seq_len}_${pred_len}" \
                    --model $model_name \
                    --data custom \
                    --features M \
                    --seq_len $seq_len \
                    --pred_len $pred_len \
                    --batch_size $bs \
                    --learning_rate $lr \
                    --enc_in 21 \
                    --dec_in 21 \
                    --c_out 21 \
                    --des 'Exp' \
                    --n_layers 1\
                    --pe_type sincos\
                    --d_model $d_model\
                    --dropout $dropout\
                    --itr 1 >${log_dir}${seq_len}_${pred_len}_${dropout}_${d_model}_bz${bs}_lr${lr}.log"

                  eval $cmd
          done
        done
      done
    done
  done
done
