export CUDA_VISIBLE_DEVICES=0
MODEL=transformer
lr=6e-5
num_layers=2
num_heads=3
python src/train.py \
    --world_size 1 \
    --total_training_samples 100000 \
    --model_type transformer \
    --model_config_path config/gpt2_tiny_wpetrain.py \
    --dataset_dir data/Composition/synthetic_multilang_rand2_train_6_100000_4_10_6 \
    --dataset_type SyntheticMultiLangRandDataset \
    --output_dir model/ \
    --batch_size 16 \
    --lr ${lr} \
    --weight_decay 0 \
    --log_interval 2048 \
    --save_interval 20000 \
    --eval_interval 20000 \
    --report_to_wandb \
   --num_hidden_layers ${num_layers} \
   --num_attention_heads ${num_heads} \