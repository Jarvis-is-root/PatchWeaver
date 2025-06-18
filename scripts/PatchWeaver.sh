export CUDA_VISIBLE_DEVICES=1
model_name=PatchWeaver
note=testall
seq_len=96
epochs=15
patience=3


for pred_len in 96 192 336 720
do

variables=7 # 7个变量
lr=1e-4
L=24
stride=24
e_layers=3
n_heads=8
d_model=256
d_ff=256
activation=gelu 
normalization=layer 
dropout=0.1 

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path  ./dataset/ETT-small/\
  --data_path ETTh1.csv \
  --model_id ETTh1_$seq_len'_'$pred_len \
  --model $model_name \
  --note $note \
  --data ETTh1 \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --enc_in $variables \
  --dec_in $variables \
  --c_out $variables \
  --des 'Exp' \
  --learning_rate $lr \
  --L $L \
  --stride $stride \
  --e_layers $e_layers \
  --n_heads $n_heads \
  --d_model $d_model \
  --d_ff $d_ff \
  --activation $activation \
  --normalization $normalization \
  --dropout $dropout \
  --itr 1 \
  --train_epochs $epochs \
  --patience $patience 
done


for pred_len in 96 192 336 720
do


variables=7 # 7个变量
lr=1e-4
L=12
stride=12
e_layers=1
n_heads=4
d_model=512
d_ff=512
activation=gelu 
normalization=layer 
dropout=0.1 

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path  ./dataset/ETT-small/\
  --data_path ETTh2.csv \
  --model_id ETTh2_$seq_len'_'$pred_len \
  --model $model_name \
  --note $note \
  --data ETTh2 \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --enc_in $variables \
  --dec_in $variables \
  --c_out $variables \
  --des 'Exp' \
  --learning_rate $lr \
  --L $L \
  --stride $stride \
  --e_layers $e_layers \
  --n_heads $n_heads \
  --d_model $d_model \
  --d_ff $d_ff \
  --activation $activation \
  --normalization $normalization \
  --dropout $dropout \
  --itr 1 \
  --train_epochs $epochs \
  --patience $patience \
  # --output_attention \ 
done


for pred_len in 96 192 336 720
do

variables=321 # 321个变量
batch_size=8
lr=1e-4
L=24
stride=24
e_layers=3
n_heads=8
d_model=512
d_ff=512
activation=gelu 
normalization=layer 
dropout=0.1 

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id ECL_$seq_len'_'$pred_len \
  --model $model_name \
  --note $note \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --enc_in $variables \
  --dec_in $variables \
  --c_out $variables \
  --des 'Exp' \
  --learning_rate $lr \
  --L $L \
  --stride $stride \
  --e_layers $e_layers \
  --n_heads $n_heads \
  --d_model $d_model \
  --d_ff $d_ff \
  --activation $activation \
  --normalization $normalization \
  --dropout $dropout \
  --itr 1 \
  --train_epochs $epochs \
  --patience $patience \
  --batch_size $batch_size \
  # --output_attention \ 
done



seq_len=720

for pred_len in 96 192 336 720
do
for e_layers in 1 2
do
for d_model in 128 256
do
variables=7 # 7个变量
lr=1e-4
L=96
stride=96
e_layers=$e_layers
n_heads=8
d_model=$d_model
d_ff=$d_model
activation=gelu 
normalization=layer 
dropout=0.1 

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path  ./dataset/ETT-small/\
  --data_path ETTm1.csv \
  --model_id ETTm1_$seq_len'_'$pred_len \
  --model $model_name \
  --note $note \
  --data ETTm1 \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --enc_in $variables \
  --dec_in $variables \
  --c_out $variables \
  --des 'Exp' \
  --learning_rate $lr \
  --L $L \
  --stride $stride \
  --e_layers $e_layers \
  --n_heads $n_heads \
  --d_model $d_model \
  --d_ff $d_ff \
  --activation $activation \
  --normalization $normalization \
  --dropout $dropout \
  --itr 1 \
  --train_epochs $epochs \
  --patience $patience \
  # --output_attention \ 
done
done
done


for pred_len in 96 192 336 720
do
for e_layers in 1 2
do
for d_model in 128 256
do
variables=7 # 7个变量
lr=1e-4
L=96
stride=96
e_layers=$e_layers
n_heads=8
d_model=$d_model
d_ff=$d_model
activation=gelu 
normalization=layer 
dropout=0.1 

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path  ./dataset/ETT-small/\
  --data_path ETTm2.csv \
  --model_id ETTm2_$seq_len'_'$pred_len \
  --model $model_name \
  --note $note \
  --data ETTm2 \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --enc_in $variables \
  --dec_in $variables \
  --c_out $variables \
  --des 'Exp' \
  --learning_rate $lr \
  --L $L \
  --stride $stride \
  --e_layers $e_layers \
  --n_heads $n_heads \
  --d_model $d_model \
  --d_ff $d_ff \
  --activation $activation \
  --normalization $normalization \
  --dropout $dropout \
  --itr 1 \
  --train_epochs $epochs \
  --patience $patience \
  # --output_attention \ 
done
done
done



for pred_len in 96 192 336 720
do
variables=21 # 21个变量
lr=1e-4
L=144
stride=144
e_layers=2
n_heads=8
d_model=128
d_ff=128
activation=gelu 
normalization=layer 
dropout=0.1 

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id weather_$seq_len'_'$pred_len \
  --model $model_name \
  --note $note \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --enc_in $variables \
  --dec_in $variables \
  --c_out $variables \
  --des 'Exp' \
  --learning_rate $lr \
  --L $L \
  --stride $stride \
  --e_layers $e_layers \
  --n_heads $n_heads \
  --d_model $d_model \
  --d_ff $d_ff \
  --activation $activation \
  --normalization $normalization \
  --dropout $dropout \
  --itr 1 \
  --train_epochs $epochs \
  --patience $patience \
  # --output_attention \ 
done
