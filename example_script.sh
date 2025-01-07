poison_path='data/4_types/sst-2/0.1_0.1_0.05_0.05'
path='results/sst-2/4_types/0.1_0.1_0.05_0.05/example'
bias_model_path='bias_models/example'
data='sst-2'
clean_path='data/clean_data/sst-2'

word_poison_path_pretrain='data/bias-only_pretrain/sst-2/badnets'
sent_poison_path_pretrain='data/bias-only_pretrain/sst-2/addsent'
syn_poison_path_pretrain='data/bias-only_pretrain/sst-2/synbkd'
style_poison_path_pretrain='data/bias-only_pretrain/sst-2/stylebkd'

# For eval
word_poison_path='data/bias-only/sst-2/badnets'
sent_poison_path='data/bias-only/sst-2/addsent'
syn_poison_path='data/bias-only/sst-2/synbkd'
style_poison_path='data/bias-only/sst-2/stylebkd'


python npoe.py \
  --data $data \
  --poison_data_path $poison_path \
  --clean_data_path $clean_path \
  --epoch 3 \
  --lr 2e-5 \
  --gpu 0 \
  --poe_alpha 1 \
  --do_PoE True \
  --small_lr 2e-5 \
  --num_hidden_layers 1 \
  --gate_hidden_layers 1 \
  --dropout_prob 0.1 \
  --result_path $path \
  --num_bias_experts 4 \
  --batch_size 8 \
  --word_poison_data_path $word_poison_path \
  --sent_poison_data_path $sent_poison_path \
  --syn_poison_data_path $syn_poison_path \
  --word_poison_path_pretrain $word_poison_path_pretrain \
  --sent_poison_path_pretrain $sent_poison_path_pretrain \
  --syn_poison_path_pretrain $syn_poison_path_pretrain \
  --freeze_bias_iters 0 \
  --epoch_pretrain 1 \
  --type test \
  --style_poison_path_pretrain $style_poison_path_pretrain \
  --style_poison_data_path $style_poison_path \
  --result_file tester \
  --do_Rdrop True \
  --rdrop_mode_2 True \
  --train_iters 1 \
