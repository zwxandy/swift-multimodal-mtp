# swift-train 环境记录

本文记录当前机器上已经跑通 `run_scripts/qwen35_mtp_sft.sh` 所用的关键环境版本与注意事项。

## 1. 结论

当前可用环境是：

- Conda 环境：`swift-train`
- Python：`3.11.0`
- GPU 驱动：`535.154.05`
- PyTorch：`2.5.1+cu121`

这套组合已经验证可以把 Qwen3.5 MTP 训练脚本跑到真正训练阶段，而不只是停在导包或数据集加载阶段。

## 2. 关键版本

### 基础运行时

- `python`: `3.11.0`
- `torch`: `2.5.1+cu121`
- `torchvision`: `0.20.1+cu121`
- `torchaudio`: `2.5.1+cu121`
- `torch.version.cuda`: `12.1`
- NVIDIA Driver: `535.154.05`
- 本机 `nvcc`: `12.8`

### Megatron / 训练核心依赖

- `ms-swift`: 当前仓库可编辑安装
- `megatron-core`: `0.15.3`
- `transformer_engine`: `1.13.0`
- `transformer_engine_torch`: `1.13.0`
- `flash-attn`: `2.8.3`

### 多模态 Qwen 依赖

- `qwen-vl-utils`: `0.0.14`
- `decord`: `0.6.0`
- `av`: 由 `qwen-vl-utils` 自动带入

### 其他关键 Python 包

- `transformers`: `5.3.0`
- `datasets`: `3.6.0`
- `modelscope`: `1.35.3`
- `peft`: `0.18.1`

## 3. 这套版本为什么这样选

### 3.1 PyTorch 必须用 cu121

这台机器的驱动版本对应的 CUDA 能力不适合直接跑 `torch 2.5.1+cu124`，否则会在 NCCL 初始化时报错：

- `CUDA driver version is insufficient for CUDA runtime version`

因此这里固定为：

- `torch==2.5.1+cu121`
- `torchvision==0.20.1+cu121`
- `torchaudio==2.5.1+cu121`

### 3.2 Transformer Engine 用 1.13.0 更稳

`transformer_engine 2.x` 在这台机器和当前 `torch 2.5.1` 组合下踩到了两类问题：

- CUDA 运行时/ABI 不匹配
- 对部分 FSDP 私有模块路径依赖不兼容

因此最终固定为：

- `transformer_engine==1.13.0`
- `transformer_engine_torch==1.13.0`

### 3.3 Qwen3.5 多模态必须补齐额外依赖

Qwen3.5 多模态链路需要：

- `qwen-vl-utils>=0.0.14`
- `decord`

否则会在模型或 processor 初始化时直接报错。

## 4. 训练脚本相关注意事项

### 4.1 数据集子集名

`LaTeX_OCR` 的子集名应为：

- `human_handwrite`

不要写成：

- `humani ndwrite`

否则会在 `datasets/modelscope` 缓存加载阶段报找不到 config。

### 4.2 attention backend

在这台机器上，`Qwen3.5 + Megatron + MTP` 直接走 `flash` 后端时，曾在真实前向中触发：

- `ValueError: No dot product attention support for the provided inputs!`

更稳的配置是：

- `--attention_backend unfused`

仓库里如果仍保留 `flash`，建议在这台机器上实际训练时改成 `unfused`。

### 4.3 flash-attn 版本提示

当前实际安装的是：

- `flash-attn==2.8.3`

日志里会提示推荐区间是：

- `>=2.1.1, <=2.6.3`

但当前环境中它已经可以正常导入和参与初始化。真正避开训练报错的关键，不是单独降级 `flash-attn`，而是将训练脚本的 `attention_backend` 调整为 `unfused`。

## 5. 代理设置

下载第三方大包时，建议先设置：

```bash
export http_proxy=http://httpproxy.glm.ai:8888
export https_proxy=http://httpproxy.glm.ai:8888
```

这对以下包的下载/编译帮助明显：

- `torch`
- `transformer_engine`
- `flash-attn`
- `qwen-vl-utils`
- `decord`

## 6. 推荐启动方式

```bash
conda activate swift-train
export http_proxy=http://httpproxy.glm.ai:8888
export https_proxy=http://httpproxy.glm.ai:8888
export PATH="/root/anaconda3/envs/swift-train/bin:$PATH"

bash run_scripts/qwen35_mtp_sft.sh
```

如果训练脚本里的注意力后端还是 `flash`，建议先改成：

```bash
--attention_backend unfused
```

## 7. 一句话总结

当前这台机器上，最关键的稳定组合是：

- `Python 3.11`
- `torch 2.5.1+cu121`
- `megatron-core 0.15.3`
- `transformer_engine 1.13.0`
- `qwen-vl-utils 0.0.14`
- `decord 0.6.0`

以及训练时优先使用：

- `--attention_backend unfused`
