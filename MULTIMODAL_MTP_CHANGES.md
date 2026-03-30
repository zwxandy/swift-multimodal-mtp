# 多模态 MTP 支持修改文档

## 背景

原始 Megatron MTP 实现只考虑纯文本场景。多模态模型在首层 embedding 阶段会把
`image/video/audio` special token 替换成视觉或音频特征，但 MTP 在构造 future-token
embedding 时默认直接调用语言模型 embedding，这会导致多模态特征丢失。

本次修复的目标是：

1. 让多模态模型可以开启 MTP 训练。
2. 保持 MTP 的核心语义不变，仍然通过上游 `_get_embeddings()` 完成 shift。
3. 让 MTP 使用和主干前向一致的多模态 embedding 注入逻辑。
4. 让多模态 special token 不参与 MTP loss。

## 宏观理解

### 纯文本 MTP 是什么

纯文本 MTP 的核心很直接：

1. 主干模型先正常算出当前位置的 `hidden_states`。
2. MTP 分支再取“未来第 k 个 token 的 embedding”作为辅助输入。
3. 用“当前位置语义 + 未来 token embedding”去学习预测更远处的文本 token。

在纯文本场景里，这个过程比较自然，因为 future-token embedding 只需要调用文本 embedding 即可，
不会丢失额外信息。

### 多模态 MTP 和纯文本 MTP 的本质区别

多模态场景下，序列里虽然仍然有 token，但其中一部分 token 只是“占位符”：

1. `image/video/audio` special token 本身并不携带真正的视觉或音频语义。
2. 真正有信息量的是 embedding 阶段注入进去的视觉/音频特征。
3. 因此，多模态 MTP 的关键不是“能不能 shift token”，而是“shift 之后拿到的 future-token embedding
   还是不是带多模态特征的那个 embedding”。

换句话说：

- 纯文本 MTP 的主要问题是时序对齐。
- 多模态 MTP 除了时序对齐，还必须保证模态特征不丢失。

### 为什么原始纯文本实现不能直接用于多模态

Megatron 原始 MTP 逻辑默认 future-token embedding 直接来自语言模型 embedding。
这在纯文本里是对的，但在多模态里会出现两个问题：

1. special token 只会映射成普通 token embedding，而不会自动变成视觉/音频特征。
2. 如果把主干输出的 `hidden_states` 直接拿来替代 future-token embedding，又会破坏 MTP 原本的训练语义。

所以，多模态 MTP 不能简单理解成“把纯文本 MTP 放开就行”，而是要解决：

1. MTP 分支里 future-token embedding 的多模态注入问题。
2. 多模态 special token 不应作为预测目标的问题。

### 这次主要改动为什么集中在 embedding 和 loss

这次没有去大改 Megatron 的 MTP 主流程，而是只修了两个关键接口：

1. **embedding 来源**
   让 MTP 在构造 future-token embedding 时，复用主干多模态模型已有的 embedding 注入逻辑。
2. **loss mask 规则**
   让 `image/video/audio` special token 在 MTP loss 中被屏蔽，不参与预测。

这样做的原因是：

1. MTP 的主干流程本身没有问题，上游的 shift 和层间结构仍然应该保留。
2. 真正导致多模态不兼容的，不是 MTP block 本身，而是 future-token embedding 的来源不对。
3. 用最小改动复用现有多模态路径，风险更低，也更符合 Megatron 原来的设计。

### 一句话总结

可以把两者的区别理解为：

- **纯文本 MTP**: 解决“未来 token 怎么预测得更远”。
- **多模态 MTP**: 在保留上面这个目标的同时，还要保证“未来 token 的输入表示仍然带有正确的多模态特征”。

## 核心设计

### 1. 不再依赖“负值占位符”

当前 Megatron 多模态链路中，视觉和音频特征是通过真实的 special token id
（例如 `image_token_id`、`video_token_id`、`audio_token_id`）定位并替换的，
而不是依赖 `input_ids < 0`。

因此，这次实现统一改为基于模型配置中的 special token id 做识别和 loss mask。

### 2. 不直接复用主干 hidden_states

MTP 需要的是“shift 后 token 的 embedding”，而不是主干 decoder 输出的 hidden states。
如果直接把主干 hidden states 当成 MTP 的 decoder input，会破坏 MTP 的训练语义。

因此，这次实现保留上游 `_get_embeddings()` 的逻辑，只把其中的 `embedding` 调用替换成
多模态感知的 embedding 闭包。

### 3. 复用已有的多模态 embedding 注入逻辑

`MultimodalGPTModel` 里原本已经通过 `_patch_word_embeddings()` 在 embedding 阶段注入视觉/音频特征。
这次没有另写一套 MTP 专用逻辑，而是复用同一个补丁路径，让主干前向和 MTP 都走同样的注入流程，
尽量减少改动并保持行为一致。

## 修改文件汇总

### 1. `swift/megatron/model/mm_gpt_model.py`

主要修改：

1. 保留多模态模型 + MTP 的入口，不再在构造时直接拦截。
2. 新增 MTP 专用 embedding 闭包。
3. 从 HF config 中收集需要在 MTP loss 中屏蔽的 special token id。

关键思路：

```python
def _get_mtp_embedding(self, multimodal_kwargs, packed_seq_params):

    def mtp_embedding(input_ids, position_ids):
        mtp_kwargs = dict(multimodal_kwargs)
        mtp_kwargs.update({'input_ids': input_ids, 'packed_seq_params': packed_seq_params})
        with self._patch_word_embeddings(mtp_kwargs):
            return self.language_model.embedding(input_ids=input_ids, position_ids=position_ids)

    return mtp_embedding
```

这样 MTP 在构造 future-token embedding 时，也会像首层前向一样把视觉/音频特征注入进去。

### 2. `swift/megatron/model/gpt_model.py`

主要修改：

1. `forward()` / `_postprocess()` 新增 `mtp_embedding` 和 `mtp_ignore_token_ids` 参数。
2. 调用 `self.mtp()` 时优先使用多模态模型传入的 `mtp_embedding`。
3. MTP loss 改为基于真实 special token id 做 mask，而不是基于负值 token。

关键思路：

```python
embedding=mtp_embedding or self.embedding
```

```python
ignore_token_mask = torch.zeros_like(input_ids, dtype=torch.bool)
for token_id in mtp_ignore_token_ids:
    ignore_token_mask |= input_ids == token_id
ignore_token_mask, _ = roll_tensor(...)
loss_mask_ = loss_mask_ & ~ignore_token_mask
```

### 3. `swift/megatron/init.py`

主要修改：

1. `_patch_mtp()` 恢复为贴近上游的调用方式。
2. 不再错误地把主干 `hidden_states` 当成 MTP 的 decoder input。
3. 调用 `_get_embeddings()` 时补上 `packed_seq_params` 透传。

关键思路：

```python
input_ids, position_ids, decoder_input, hidden_states = self._get_embeddings(
    input_ids=input_ids,
    position_ids=position_ids,
    embedding=embedding,
    hidden_states=hidden_states,
    packed_seq_params=packed_seq_params,
)
```

这样可以保证 MTP 继续沿用 Megatron 原本的 shift 逻辑，只把 embedding 计算替换为多模态兼容版本。

## 使用示例

```bash
CUDA_VISIBLE_DEVICES=3 \
NPROC_PER_NODE=1 \
megatron sft \
    --model /workspace/wenxuan/ckpt/Qwen3.5-4B \
    --save_safetensors true \
    --mtp_num_layers 1 \
    --dataset 'AI-ModelScope/LaTeX_OCR:human_handwrite#2000' \
    --tuner_type lora \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --output_dir output/Qwen3.5-4B-mtp-multimodal
```

## 注意事项

1. **special token 识别方式**: 当前实现使用 HF config 中的 `image_token_id`、`video_token_id`、
   `audio_token_id` 来识别多模态输入位。
2. **MTP 只预测文本**: 多模态 special token 在 MTP loss 中会被 mask，不作为预测目标。
3. **与主干一致**: MTP 的 future-token embedding 复用了主干 embedding 的多模态注入路径。
4. **最小改动原则**: 没有改动 MTP 的主流程结构，只替换了 embedding 来源和 loss mask 条件。

## 修改标签

所有核心修复处都带有 `[multimodal mtp]` 注释标签，方便后续检索和维护。
