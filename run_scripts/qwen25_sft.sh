# 显存占用：22GB
CUDA_VISIBLE_DEVICES=3 \
swift sft \
    --model /cloud/oss_checkpoints/Qwen2.5-7B-Instruct \
    --tuner_type lora \
    --dataset 'swift/Qwen3-SFT-Mixin#2000' \
              'swift/self-cognition:qwen3#600' \
    --load_from_cache_file true \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-4 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --save_steps 50 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --max_length 2048 \
    --output_dir output \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --model_author swift \
    --model_name swift-robot