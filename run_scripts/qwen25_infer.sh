CUDA_VISIBLE_DEVICES=3 \
swift infer \
    --model /cloud/oss_checkpoints/Qwen2.5-7B-Instruct \
    --infer_backend vllm \
    --stream true \
    --max_new_tokens 2048 \
    --vllm_max_model_len 8192 \
    --temperature 0.0 \
    --adapter output/v0-20260326-112133/checkpoint-102