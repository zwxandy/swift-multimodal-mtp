CUDA_VISIBLE_DEVICES=3 \
swift infer \
    --model /workspace/wenxuan/ms-swift/run_scripts/megatron_output/Qwen3.5-4B/v11-20260330-203929/checkpoint-187-merged \
    --vllm_tensor_parallel_size 4 \
    --infer_backend vllm \
    --vllm_max_model_len 8192 \
    --vllm_gpu_memory_utilization 0.9 \
    --vllm_speculative_config '{"method":"qwen3_next_mtp","num_speculative_tokens":2}' \
    --max_new_tokens 2048
