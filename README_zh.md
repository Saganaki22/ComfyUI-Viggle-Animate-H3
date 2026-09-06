# ComfyUI-Viggle-Animate-H3

**[English](README.md) | [中文](README_zh.md)**

<img width="731" height="468" alt="Screenshot 2026-09-05 214633" src="https://github.com/user-attachments/assets/1b65c73d-555e-4097-a2de-7ae2f5e6851f" />

**[Viggle-Animate](https://huggingface.co/Viggle/Viggle-Animate)** 的 ComfyUI 节点 —— 这是对
MiniMax-H3 `ref2va` transformer 的 33.1 B 全量微调,用于**视频角色替换**:输入一段驱动视频和
一张参考图,即可将视频中的表演者替换为参考图中的角色。动作、运镜、节奏、背景和光线来自视频;
身份特征来自图片。

无需文本编码器、无需提示词:条件输入是 Viggle 团队用 Qwen3-VL 预先计算一次的 362 token
冻结嵌入(`assets/fixed_prompt.txt`),每次渲染完全相同。采样器经过 DMD2 蒸馏 ——
**4 步(3 次前向传播)**,即使在消费级硬件上,124 帧片段也很快。

## 节点

| 节点 | 功能 |
|---|---|
| **Load Text Conditioning (Viggle)** | 从 `models/text_cond/` 下拉加载冻结文本条件 |
| **Viggle-Animate Conditioning (H3)** | 构建条件 + AV latent:视频优先的参考顺序,两个参考均按驱动视频短边嵌套 —— 即微调训练时使用的布局 |

其余均使用 ComfyUI 核心节点:**Load Diffusion Model**、**Load LoRA (Model Only)**、
**ModelSamplingMiniMaxH3**(shift_video 3.0)、**KSampler**、**VAE Decode**、**Save Video**。

| 驱动视频 | 参考图 | 输出 |
|:---:|:---:|:---:|
| <video src="https://github.com/user-attachments/assets/2529857c-2667-4641-9d2e-5dcb3c03913d" controls muted></video> | <img src="https://github.com/user-attachments/assets/f6adf969-03d5-4a58-bd30-5ec2d0bc604b" width="300"> | <video src="https://github.com/user-attachments/assets/deedde68-80de-47d2-9e1f-7eaa3bc35457" controls muted></video> |

| 示例 1 | 示例 2 | 示例 3 | 示例 4 |
|:---:|:---:|:---:|:---:|
| <video src="https://github.com/user-attachments/assets/c5198b4c-9544-4e5a-bd1e-83ae4477bb0b" controls muted></video> | <video src="https://github.com/user-attachments/assets/afb74de1-3d3d-42ae-9885-1d5b1c6e86af" controls muted></video> | <video src="https://github.com/user-attachments/assets/5ddf1bb1-e744-406b-8b9f-2b729e4ecc4b" controls muted></video> | <video src="https://github.com/user-attachments/assets/8d02389d-67ca-46f0-b9e3-78994d944a90" controls muted></video> |


### euler / beta - 6 步

## 安装

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/Saganaki22/ComfyUI-Viggle-Animate-H3
```

### 权重 —— 已转换的 ComfyUI 原生格式:[drbaph/Viggle-Animate-ComfyUI](https://huggingface.co/drbaph/Viggle-Animate-ComfyUI)

**扩散模型** → `ComfyUI/models/diffusion_models/`

| 文件 | 大小 | 说明 |
|---|---|---|
| [minimax_h3_ref2va_viggle_bf16.safetensors](https://huggingface.co/drbaph/Viggle-Animate-ComfyUI/resolve/main/diffusion_models/minimax_h3_ref2va_viggle_bf16.safetensors) | 66.3 GB | 全精度 —— 需要 ≥ 96 GB 显存或 offload |
| [minimax_h3_ref2va_viggle_int8_convrot.safetensors](https://huggingface.co/drbaph/Viggle-Animate-ComfyUI/resolve/main/diffusion_models/minimax_h3_ref2va_viggle_int8_convrot.safetensors) | 47 GB | int8 权重(convrot 量化) |
| [minimax_h3_ref2va_viggle_pruned_int8_convrot.safetensors](https://huggingface.co/drbaph/Viggle-Animate-ComfyUI/resolve/main/diffusion_models/minimax_h3_ref2va_viggle_pruned_int8_convrot.safetensors) | 21 GB | + 低秩 `adaln_proj` —— **适配 32 GB 显卡,推荐** |

**DMD LoRA** → `ComfyUI/models/loras/`(将采样压缩到 4 步的蒸馏增量 ——
必需;它是作用于*微调后* transformer 的增量,不是原版 MiniMax 的)

| 文件 | 大小 | 说明 |
|---|---|---|
| [viggle_animate_dmd_lora.safetensors](https://huggingface.co/drbaph/Viggle-Animate-ComfyUI/resolve/main/loras/viggle_animate_dmd_lora.safetensors) | 3.8 GB | 原始 rank 128,无损转换 |
| [viggle_animate_dmd_lora_r64.safetensors](https://huggingface.co/drbaph/Viggle-Animate-ComfyUI/resolve/main/loras/viggle_animate_dmd_lora_r64.safetensors) | 0.94 GB | SVD 截断,保留 99.26% 谱能量 |

**冻结文本条件** → `ComfyUI/models/text_cond/`

| 文件 | 说明 |
|---|---|
| [fixed_embed_fwd_anyframe.safetensors](https://huggingface.co/drbaph/Viggle-Animate-ComfyUI/resolve/main/text_cond/fixed_embed_fwd_anyframe.safetensors) | 362 token 冻结嵌入 —— 完全替代文本编码器 |

**VAE**(来自基础模型,非微调仓库)→ `ComfyUI/models/vae/`
来自 [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3):
`vae/minimax_h3_video_vae_fp16.safetensors` 和 `vae/minimax_h3_audio_vae_fp32.safetensors`

## 工作流配置

```
Load Diffusion Model(viggle pruned_int8_convrot)
  -> Load LoRA (Model Only)(viggle_animate_dmd_lora,strength 1)
  -> ModelSamplingMiniMaxH3(shift_video 3.0,shift_audio 3.0)
  -> KSampler(4-8 步,cfg 1.0)

Load Video(24 fps,frame_load_cap = length) --+
Load Image(参考图) ---------------------------+--> Viggle-Animate Conditioning (H3) --+
Load VAE(minimax_h3 video VAE) --------------+                                      |
Load Text Conditioning (Viggle) --------------+                            KSampler --+--> VAE Decode -> Save Video
```

关键设置:

- **4-8 步,cfg 1.0** —— 4 步是评测使用的最佳工作点;步数更多会趋向过度锐化
- **shift_video 3.0** —— 基础默认值 12 不适用于蒸馏模型
- **驱动视频长度 = 输出长度**(将 Load Video 的 frame_load_cap 设为 `length`,例如 124 帧 @ 24 fps)
- **width/height 设为 0** = 继承驱动视频的几何尺寸(输出两边必须是 32 的倍数);
  显式设置时即为输出画布(每轴取整到 32)。测试画布范围:**0.4–0.98 百万像素**
- 可与 Comfy Kitchen 和 block sparse attention 补丁叠加使用

## 已知局限

- **重新入镜时身份漂移**:当主体离开镜头后重新入镜,画面会趋向驱动视频中的原始外观,
  而非参考图。当主体大幅偏离参考姿态或做出剧烈动作(如后空翻)时同样如此 ——
  与参考图的姿态差距越大,身份保持越弱。

## 链接

- 原始模型 + 推理代码:[huggingface.co/Viggle/Viggle-Animate](https://huggingface.co/Viggle/Viggle-Animate) · [viggle.ai](https://viggle.ai)
- 基础模型:[huggingface.co/MiniMaxAI/MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3)
- Comfy 重新打包的基础 VAE:[huggingface.co/Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3)
- 转换后的 ComfyUI 权重、量化与 LoRA:[huggingface.co/drbaph/Viggle-Animate-ComfyUI](https://huggingface.co/drbaph/Viggle-Animate-ComfyUI)

## 引用

如果在已发表的工作中使用该模型,请引用原文:

```bibtex
@misc{viggle2026animate,
  title  = {Viggle-Animate: Character Replacement in Video from a Single Repainted Frame},
  author = {Viggle Research},
  year   = {2026},
  url    = {https://huggingface.co/Viggle/Viggle-Animate}
}
```

## 许可与负责任使用

- **权重**是 MiniMax H3 的模型衍生品 ——
  [MiniMax H3 Community License Agreement](https://huggingface.co/MiniMaxAI/MiniMax-H3)
  适用于它们(在再分发或用于产品之前请阅读)。这包括上面链接的转换/量化变体。
- 本**节点包**采用 Apache 2.0 许可(见 `LICENSE`)。
- 该模型可以将人物放入其未参与拍摄的视频中;身份来自你提供的图片。
  请勿在未获同意的人身上使用,并将生成内容标注为 AI 生成(见原始仓库的 intended-use 部分)。
