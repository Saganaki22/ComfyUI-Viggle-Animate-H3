# ComfyUI-Viggle-Animate-H3

**[English](README.md) | [中文](README_zh.md)**

<img width="731" height="468" alt="Screenshot 2026-09-05 214633" src="https://github.com/user-attachments/assets/1b65c73d-555e-4097-a2de-7ae2f5e6851f" />

**[Viggle-Animate](https://huggingface.co/Viggle/Viggle-Animate)** 的 ComfyUI 节点 —— 这是对 MiniMax-H3 `ref2va` transformer 的 33.1 B 全量微调，用于**视频角色替换**：输入一段驱动视频和一张参考图，即可将视频中的表演者重新渲染为参考图中的角色。动作、运镜、节奏、背景和光线来自视频；身份特征来自图片。

无需文本编码器、无需提示词：条件输入是 Viggle 团队使用 Qwen3-VL 预先计算一次的 362 token 冻结嵌入（`assets/fixed_prompt.txt`），每次渲染完全相同。

采样器经过 DMD2 蒸馏，推荐使用 **4–8 步**，其中 **6 步是速度与质量之间的最佳平衡点**。仓库内附带的工作流同时包含普通 scheduler 配置以及**根据上游采样公式推导出的手动 sigma 配置**，可直接使用 ComfyUI 自带的 **ManualSigmas** 节点。

需要注意：手动 sigma 中所谓的“步数”指的是 **sigma 点数量，并包含最后的 `0.0`**。因此：

```text
4 个 sigma 点 = 4 步 = 3 次模型前向传播
6 个 sigma 点 = 6 步 = 5 次模型前向传播
8 个 sigma 点 = 8 步 = 7 次模型前向传播
```

也就是说，4-step 并不代表 4 次模型推理，而是 4 个 sigma 点，其中最后一个 `0.0` 是轨迹终点，因此实际只执行 3 次 forward。

## 节点

| 节点                                   | 功能                                                                                                        |
| ------------------------------------ | --------------------------------------------------------------------------------------------------------- |
| **Load Text Conditioning (Viggle)**  | 从 `models/text_cond/` 下拉加载冻结文本条件                                                                          |
| **Viggle-Animate Conditioning (H3)** | 构建 conditioning + AV latent：采用视频优先的参考顺序，并将两个参考按画布短边嵌套；默认使用驱动视频的短边尺寸，也可通过 width/height 覆盖 —— 与微调训练时使用的布局一致 |

其余均使用 ComfyUI 核心节点：**Load Diffusion Model**、**Load LoRA (Model Only)**、**ModelSamplingMiniMaxH3**（shift_video 3.0）、**KSampler**、**VAE Decode**、**Save Video**。使用手动 sigma 工作流时，还会使用 ComfyUI 自带的 **ManualSigmas** 节点。

|                                                         驱动视频                                                         |                                                   参考图                                                   |                                                          输出                                                          |
| :------------------------------------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------------------: | :------------------------------------------------------------------------------------------------------------------: |
| <video src="https://github.com/user-attachments/assets/2529857c-2667-4641-9d2e-5dcb3c03913d" controls muted></video> | <img src="https://github.com/user-attachments/assets/f6adf969-03d5-4a58-bd30-5ec2d0bc604b" width="300"> | <video src="https://github.com/user-attachments/assets/deedde68-80de-47d2-9e1f-7eaa3bc35457" controls muted></video> |

|                                                         示例 1                                                         |                                                         示例 2                                                         |                                                         示例 3                                                         |                                                         示例 4                                                         |
| :------------------------------------------------------------------------------------------------------------------: | :------------------------------------------------------------------------------------------------------------------: | :------------------------------------------------------------------------------------------------------------------: | :------------------------------------------------------------------------------------------------------------------: |
| <video src="https://github.com/user-attachments/assets/c5198b4c-9544-4e5a-bd1e-83ae4477bb0b" controls muted></video> | <video src="https://github.com/user-attachments/assets/afb74de1-3d3d-42ae-9885-1d5b1c6e86af" controls muted></video> | <video src="https://github.com/user-attachments/assets/5ddf1bb1-e744-406b-8b9f-2b729e4ecc4b" controls muted></video> | <video src="https://github.com/user-attachments/assets/8d02389d-67ca-46f0-b9e3-78994d944a90" controls muted></video> |

### euler / beta - 6 步

## 安装

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/Saganaki22/ComfyUI-Viggle-Animate-H3
```

## 模型下载

### 权重 —— 已转换的 ComfyUI 原生格式

[drbaph/Viggle-Animate-ComfyUI](https://huggingface.co/drbaph/Viggle-Animate-ComfyUI)

### 扩散模型

放入：

```text
ComfyUI/models/diffusion_models/
```

| 文件                                                                                                                                                                                                      |      大小 | 说明                                          |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------: | ------------------------------------------- |
| [minimax_h3_ref2va_viggle_bf16.safetensors](https://huggingface.co/drbaph/Viggle-Animate-ComfyUI/resolve/main/diffusion_models/minimax_h3_ref2va_viggle_bf16.safetensors)                               | 66.3 GB | BF16，全精度版本，质量最高；通常需要 ≥96 GB 显存或进行 offload   |
| [minimax_h3_ref2va_viggle_int8_convrot.safetensors](https://huggingface.co/drbaph/Viggle-Animate-ComfyUI/resolve/main/diffusion_models/minimax_h3_ref2va_viggle_int8_convrot.safetensors)               |   47 GB | int8 权重（convrot 量化）                         |
| [minimax_h3_ref2va_viggle_pruned_int8_convrot.safetensors](https://huggingface.co/drbaph/Viggle-Animate-ComfyUI/resolve/main/diffusion_models/minimax_h3_ref2va_viggle_pruned_int8_convrot.safetensors) |   21 GB | + 低秩 `adaln_proj`，显存友好版本；**适配 32 GB 显卡，推荐** |

### DMD LoRA

放入：

```text
ComfyUI/models/loras/
```

DMD LoRA 是低步数采样所使用的蒸馏增量。它作用于**已经完成 Viggle 微调的 transformer**，不是作用于原版 MiniMax-H3。

| 文件                                                                                                                                                         |      大小 | 说明                         |
| ---------------------------------------------------------------------------------------------------------------------------------------------------------- | ------: | -------------------------- |
| [viggle_animate_dmd_lora.safetensors](https://huggingface.co/drbaph/Viggle-Animate-ComfyUI/resolve/main/loras/viggle_animate_dmd_lora.safetensors)         |  3.8 GB | 原始 full-rank / rank 128 版本 |
| [viggle_animate_dmd_lora_r64.safetensors](https://huggingface.co/drbaph/Viggle-Animate-ComfyUI/resolve/main/loras/viggle_animate_dmd_lora_r64.safetensors) | 0.94 GB | rank 64，推荐；体积更小            |

### 冻结文本条件

放入：

```text
ComfyUI/models/text_cond/
```

| 文件                                                                                                                                                       | 说明                                                                    |
| -------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------- |
| [fixed_embed_fwd_anyframe.safetensors](https://huggingface.co/drbaph/Viggle-Animate-ComfyUI/resolve/main/text_cond/fixed_embed_fwd_anyframe.safetensors) | 362 token 冻结嵌入 —— 完全替代文本编码器，使用 **Load Text Conditioning (Viggle)** 加载 |

### VAE

放入：

```text
ComfyUI/models/vae/
```

可选：

* [minimax_h3_video_vae_int8_convrot.safetensors](https://huggingface.co/Kijai/MiniMax-H3-experimental/resolve/main/minimax_h3_video_vae_int8_convrot.safetensors)（3.17 GB，低显存）
* [minimax_h3_video_vae_fp16.safetensors](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/vae/minimax_h3_video_vae_fp16.safetensors)（5.21 GB）

## 模型目录结构

```text
📂 ComfyUI/
├── 📂 models/
│   ├── 📂 diffusion_models/
│   │   └── minimax_h3_ref2va_viggle_pruned_int8_convrot.safetensors
│   ├── 📂 loras/
│   │   └── viggle_animate_dmd_lora_r64.safetensors
│   ├── 📂 text_cond/
│   │   └── fixed_embed_fwd_anyframe.safetensors
│   └── 📂 vae/
│       └── minimax_h3_video_vae_int8_convrot.safetensors
```

## 工作流配置

```text
Load Diffusion Model (viggle pruned_int8_convrot)
  -> Load LoRA (Model Only) (viggle_animate_dmd_lora_r64, strength 1)
  -> ModelSamplingMiniMaxH3 (shift_video 3.0)
  -> KSampler / SamplerCustom

Load Video (24 fps, frame_load_cap = length) --+
Load Image (参考图) ----------------------------+--> Viggle-Animate Conditioning (H3) --+
Load VAE (MiniMax-H3 video VAE) ---------------+                                       |
Load Text Conditioning (Viggle) ---------------+                              Sampler --+--> VAE Decode -> Save Video
```

关键设置：

* 推荐采样范围为 **4–8 步**
* **6 步是推荐的速度 / 质量甜点位**
* `cfg 1.0`
* **ModelSamplingMiniMaxH3：`shift_video 3.0`**
* 驱动视频长度应与输出长度一致：将 **Load Video** 的 `frame_load_cap` 设置为 conditioning 节点的 `length`，例如 `124`
* **Load Video** 建议使用 `force_rate = 24`
* `width = 0` / `height = 0` 时，默认继承驱动视频尺寸
* 手动指定 width / height 时，每个轴会取整到 32 的倍数
* 测试画布范围约为 **0.4–0.98 百万像素**
* 可与 **Comfy Kitchen** 和 **block sparse attention** 补丁叠加使用

## 采样器与 Manual Sigmas

### 普通 Scheduler

对于 ComfyUI 的普通 scheduler 工作流，推荐使用 **6–8 步**。

可使用：

```text
simple
beta
normal
bong_tangent
```

也测试过：

```text
euler
er_sde
exp_heun_2_x0
lcm
```

其中 **6 步通常是速度与质量之间最好的平衡点**。

如果更看重速度，可以尝试 4 步；如果希望进一步提高稳定性或质量，可以尝试 8 步。

## Manual Sigmas

仓库中的工作流还包含使用 ComfyUI 自带 **ManualSigmas** 节点的版本。

这些 sigma 数值并不是任意手调的，而是**根据上游采样公式推导得到的**。

这里最重要的一点是：

> ManualSigmas 中的“step 数”指 sigma 点数量，其中包含最后的 `0.0`。

因此：

```text
4 steps = 4 个 sigma 点 = 3 次 forward
6 steps = 6 个 sigma 点 = 5 次 forward
8 steps = 8 个 sigma 点 = 7 次 forward
```

原因是相邻两个 sigma 点之间才对应一次采样更新。

例如：

```text
sigma_0 -> sigma_1
sigma_1 -> sigma_2
sigma_2 -> sigma_3
```

4 个 sigma 点只有 3 个区间，因此只需要 **3 次模型前向传播**。

### 4-step Manual Sigmas

**4 个 sigma 点 / 3 次 forward**

在 ComfyUI 自带的 **ManualSigmas** 节点中输入：

```text
1.0, 0.8571428571428571, 0.6, 0.0
```

对应关系：

```text
4 sigma points
→ 3 个采样区间
→ 3 次模型 forward
```

这是提供的最快手动 sigma 配置。

### 6-step Manual Sigmas

**6 个 sigma 点 / 5 次 forward**

输入：

```text
1.0, 0.9230769230769231, 0.8181818181818182, 0.6666666666666666, 0.42857142857142855, 0.0
```

对应：

```text
6 sigma points
→ 5 个采样区间
→ 5 次模型 forward
```

**推荐优先使用这一组。**

它通常是速度和质量之间最好的平衡点。

### 8-step Manual Sigmas

**8 个 sigma 点 / 7 次 forward**

输入：

```text
1.0, 0.9473684210526315, 0.8823529411764706, 0.8, 0.6923076923076923, 0.5454545454545454, 0.3333333333333333, 0.0
```

对应：

```text
8 sigma points
→ 7 个采样区间
→ 7 次模型 forward
```

这组会执行更多模型计算，更偏向质量和稳定性。

### 应该选哪个？

建议从 **6 步**开始：

```text
4 steps / 3 forwards → 最快
6 steps / 5 forwards → 推荐，速度 / 质量最佳平衡
8 steps / 7 forwards → 更偏向质量与稳定性
```

如果使用普通 scheduler：

```text
6–8 steps
simple / beta / normal / bong_tangent
```

是比较推荐的起点。

如果使用仓库内提供的 **ManualSigmas 工作流**，则直接使用上面的 4 / 6 / 8 sigma 配置即可。

这些手动 sigma 已经包含在仓库的工作流中。

## 已知局限

* **重新入镜时身份漂移**：当主体离开镜头后重新入镜时，重新出现的主体可能逐渐趋向驱动视频中的原始外观，而不是参考图中的角色。
* **大幅动作时身份保持下降**：当主体姿态与参考图差异很大，或者进行突然、剧烈的动作（例如后空翻）时，身份保持能力会减弱。与参考图姿态差异越大，reference identity 的约束通常越弱。
* **口型同步限制**：生成角色不会稳定地与驱动视频中的口型保持同步。
* **参考图兼容性**：如果输出中的角色无法很好地保持参考图身份，建议让参考图中的人物姿态 / 站姿尽可能接近驱动视频中的人物，并尽可能保持相似背景。遇到明显 identity drift 时，建议将生成分辨率控制在 **0.4–0.6 MP**，并尝试 **LCM 或 normal + 6–8 步**。

## 链接

* 原始模型 + 推理代码：[huggingface.co/Viggle/Viggle-Animate](https://huggingface.co/Viggle/Viggle-Animate) · [viggle.ai](https://viggle.ai)
* 基础模型：[huggingface.co/MiniMaxAI/MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3)
* Comfy 重新打包的基础 VAE：[huggingface.co/Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3)
* 转换后的 ComfyUI 权重、量化与 LoRA：[huggingface.co/drbaph/Viggle-Animate-ComfyUI](https://huggingface.co/drbaph/Viggle-Animate-ComfyUI)

## 引用

如果在已发表的工作中使用该模型，请引用原始项目：

```bibtex
@misc{viggle2026animate,
  title  = {Viggle-Animate: Character Replacement in Video from a Single Repainted Frame},
  author = {Viggle Research},
  year   = {2026},
  url    = {https://huggingface.co/Viggle/Viggle-Animate}
}
```

## 许可与负责任使用

* **权重**是 MiniMax H3 的模型衍生品 —— [MiniMax H3 Community License Agreement](https://huggingface.co/MiniMaxAI/MiniMax-H3) 适用于这些权重。在重新分发或将其用于产品之前，请先阅读相关许可条款。这也包括上面链接的转换版和量化版权重。
* 本**节点包**采用 Apache 2.0 许可（见 `LICENSE`）。
* 该模型可以将人物身份替换进其未参与拍摄的视频中；身份来源于你提供的参考图片。请勿在未获得本人同意的情况下使用他人身份，并建议明确标注生成内容为 AI 生成内容（参见原始仓库的 intended-use 部分）。

## 问题反馈

* Issues：[ComfyUI-Viggle-Animate-H3/issues](https://github.com/Saganaki22/ComfyUI-Viggle-Animate-H3/issues)
