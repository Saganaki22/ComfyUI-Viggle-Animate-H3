# ComfyUI-Viggle-Animate-H3

**[English](README.md) | [中文](README_zh.md)**

<img width="1273" height="817" alt="image" src="https://github.com/user-attachments/assets/f181daf9-3d93-46e8-977a-6c1f15f4b303" />

**[Viggle-Animate](https://huggingface.co/Viggle/Viggle-Animate)** 的 ComfyUI 节点 —— 这是对 MiniMax-H3 `ref2va` transformer 的 33.1 B 全量微调，用于**视频角色替换**：输入一段驱动视频和一张参考图，即可将视频中的表演者重新渲染为参考图中的角色。动作、运镜、节奏、背景和光线来自视频；身份特征来自图片。

无需文本编码器、无需提示词：条件输入是 Viggle 团队使用 Qwen3-VL 预先计算一次的 362 token 冻结嵌入（`assets/fixed_prompt.txt`），每次渲染完全相同。

采样器经过 DMD2 蒸馏，上游基准为 **4 个 sigma 点，即 3 次 Euler 更新**。仓库内附带的工作流同时包含普通 scheduler 配置以及**根据上游采样公式推导出的手动 sigma 配置**，可直接使用 ComfyUI 自带的 **ManualSigmas** 节点。

这些手动预设搭配 Euler 和 BasicGuider / CFG 1.0 时，**sigma 点数包含最后的 `0.0`，实际采样更新次数（模型前向传播次数）比点数少一次**：

```text
4 个 sigma 点 = 3 次采样更新 / 模型前向传播：快速
6 个 sigma 点 = 5 次采样更新 / 模型前向传播：平衡
8 个 sigma 点 = 7 次采样更新 / 模型前向传播：画质优先（可能过度锐化）
```

也就是说，4-step 并不代表 4 次模型推理，而是 4 个 sigma 点，其中最后一个 `0.0` 是轨迹终点，因此实际只执行 3 次 forward。

## 1.3.2 更新

- 修复非网格帧数下末块动作参考与生成目标的潜变量长度不一致：VAE 编码前填充末尾参考，使两者长度匹配。维护者的 289 帧测试确认末尾漂移已解决。
- 恢复完整长度、末尾对齐的最后一个窗口，保留源视频结尾。
- 新增默认 `five_frame_anchor` 模式：解码并重新编码 5 帧作为续接锚点，已接受的输出不会被后续重叠覆盖。可切换 `latent_overlap` 对比旧模式。
- Chunked Sampler 在解码后裁掉网格填充，输出帧数等于加载的源帧数。高级循环输出 LATENT，外部解码仍包含网格填充。
- 更新示例工作流和检查点处理。更新后重启 ComfyUI 并刷新浏览器；旧循环工作流使用锚点模式时，需将 H3 VAE 接到 **Sample Chunk**。代码更新会使自动恢复失效，已有文件仍可读取。

## 1.3.0 更新

新增分窗口条件节点和 **Viggle Chunked Sampler**，支持长视频分块生成、潜变量重叠传递、分块缓存复用，以及通过种子覆盖重试某一段。新增从上游 shift-3 公式推导的 **4、6、8 点自定义 sigma 预设**，分别适合**快速、平衡、画质优先（可能过度锐化）**。

## 节点

本节点包还包含四个循环节点，支持磁盘检查点、外部 VAE 解码和实时分块进度。
接线、恢复和首次测试步骤见[长视频生成指南](docs/long_video_guide_zh.md)。

| 节点 | 功能 |
|---|---|
| **Load Text Conditioning (Viggle)** | 从 `models/text_cond/` 下拉加载冻结文本条件 |
| **Viggle-Animate Conditioning (H3)** | 构建条件 + AV latent:视频优先的参考顺序,两个参考均按驱动视频短边嵌套 —— 即微调训练时使用的布局 |
| **Viggle-Animate Conditioning (H3, Windowed)** | 将驱动视频划分为重叠窗口，并为每块构建参考条件；输出 `cond_set` 接分块采样器，`guider_positive` 接 guider |
| **Viggle Chunked Sampler** | 逐块采样并保留上一块的重叠内容，复用符合条件的缓存，最后统一解码；输出视频帧 `frames` 和分块信息 `chunk_map` |
| **Viggle Chunk Loop Start** | 创建运行目录并按分块计划启动循环；`initial_state` 留空 |
| **Viggle Sample Chunk** | 采样并保存当前块；输出 LATENT、循环状态、保存文件名前缀，显示实时进度 |
| **Viggle Chunk Loop End** | 等待当前块解码及所连接的保存节点完成，再进入下一块 |
| **Viggle Assemble Chunk Latents** | 加载单块或拼接匹配的检查点链，输出 LATENT 供最终解码，并报告完整/部分状态 |

### 该用哪种工作流？

下载 JSON 工作流，或将对应 PNG 拖入 ComfyUI：

| 工作流 | JSON | PNG |
|---|---|---|
| 单镜头（v1.2.0） | [JSON](example_workflows/viggle-animate-h3_workflow-v1.2.0.json) | [PNG](example_workflows/viggle-animate-h3_workflow-v1.2.0.png) |
| Chunked Sampler：内存缓存，最后统一解码 | [JSON](example_workflows/viggle-animate-h3_workflow-chunked-sampler.json) | [PNG](example_workflows/viggle-animate-h3_workflow-chunked-sampler.png) |
| Long Video Advanced：循环、磁盘检查点、外部解码 | [JSON](example_workflows/viggle-animate-h3_workflow-long-video-advanced.json) | [PNG](example_workflows/viggle-animate-h3_workflow-long-video-advanced.png) |

| 用途 | 选择 |
|---|---|
| 单镜头，通常为 124 帧（24 fps 下约 5.2 秒） | 原始 Conditioning → 核心采样器 → VAE Decode → 保存 |
| 长片段，希望接线简单、只在最后解码一次 | Windowed Conditioning → **Viggle Chunked Sampler**；缓存仅在内存中 |
| 长时间生成、每块预览，或需要取消/重启后恢复 | Windowed Conditioning → **Start / Sample Chunk / End → Assemble → VAE Decode**；检查点保存在磁盘 |

即使只有两块，只要不想丢失已采样的进度，也可以选择循环工作流。
只需放置一个 Sample Chunk，循环次数由计划自动计算。
从 **124 帧窗口、22 帧重叠、24 fps** 和下方四点 sigma 基准开始。
**Chunked Sampler** 内部解码；**Sample Chunk** 输出潜变量，需要外部 VAE Decode。

模型加载和采样控制使用 ComfyUI 核心节点：**Load Diffusion Model**、**Load LoRA (Model Only)**、
**ModelSamplingMiniMaxH3**（视频/音频 shift 均为 3.0）、**BasicGuider**、**KSamplerSelect** 和 **ManualSigmas**。
也可使用 KJNodes 的 **CustomSigmas** 输入下方调度。分块采样器内部已完成解码，`frames` 直接连接视频保存节点。

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

更新后重启 ComfyUI 并刷新浏览器，以加载实时进度扩展。

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

单镜头工作流的接线（核心采样路径）：

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

- **上游采样基准：** 使用 **ManualSigmas** 输入 `1.0, 0.8571428571428571, 0.6, 0.0`，搭配 **Euler**、**BasicGuider**（或 CFG 1.0），以及原始 `viggle_animate_dmd_lora.safetensors`，强度 1.0。将 sigma 输出连接到 SamplerCustomAdvanced 或 Viggle Chunked Sampler。四个 sigma 点对应 **3 次模型计算**，即上游所称的“4 步”。
- **ModelSamplingMiniMaxH3** 的视频、音频 shift 都保留 **3.0**。它不会再次变换手动输入的 sigma 列表；不要在列表后再接 sigma 变换节点。
- **KJNodes CustomSigmas：** 上述四个值应配合 `interpolate_to_steps = 3`。设为 4 会对包含零的序列做对数插值，得到以 `0, 0` 结尾的调度；Euler 随后除以零，产生 NaN，导致最终视频和后续分块变黑。分块采样器现在会在渲染前拒绝这种无效调度。
- **Load Video：** 使用 `force_rate = 24`。单段生成的 `length` 是上限：保持 124 时，56 帧输入会自动生成 56 帧。不符合网格的输入向上对齐到下一个 `17k+5` 长度，不丢弃参考帧。分窗口生成时加载所需的完整视频；VHS 的 `frame_load_cap = 0` 表示加载全部帧。
- **width/height 设为 0** 表示继承驱动视频尺寸；显式设置时决定输出画布，每轴取整到 32。测试画布范围：**0.4–1.2 百万像素** —— 1.2 MP 下画质依然可靠，但驱动视频和参考图必须足够清晰，不能有像素化。
- **4 点 / 3 次更新适合快速生成，6 点 / 5 次更新兼顾速度与画质，8 点 / 7 次更新偏重画质（可能过度锐化）**。请结合所用的 ComfyUI 转换模型和量化版本比较效果。
- 已测试的采样器/调度包括 `euler`、`er_sde`、`exp_heun_2_x0`、`lcm` / `simple`、`normal`、`beta`、`bong_tangent`，配合 CFG 1.0 与 shift 3.0。
- 仓库内工作流同时包含普通 scheduler 配置和根据上游公式推导的手动 sigma 配置（经 KJNodes **CustomSigmas** 内置在工作流文件中）。
- 其他采样器、调度和 rank-64 LoRA 属于可尝试的替代方案。旧版示例工作流使用的 LCM / bong_tangent 八步配置与上游基准不同。
- 可叠加 Comfy Kitchen 和 block sparse attention 补丁。

## 自定义 sigma 预设（4、6 或 8 点）

将下方任一列表粘贴到 **ManualSigmas** 或 KJNodes **CustomSigmas**，并将其 `SIGMAS` 输出连接采样器。公式为 `sigma = 3*t / (1 + 2*t)`，其中 `t` 从 1 到 0 等间隔取值。四点预设匹配上游基准，六点和八点预设按同一规律扩展。

| 建议的工作流节点标题 | sigma 点数（包含末尾零） | 采样更新次数 / KJNodes `interpolate_to_steps` |
|---|---|---|
| **Viggle DMD — 3 Steps (Upstream “4-Step”)** | 4 | **3** |
| **Viggle — 5 Steps (6 Sigma Points)** | 6 | **5** |
| **Viggle — 7 Steps (8 Sigma Points)** | 8 | **7** |

**4 点 / 3 次 Euler 更新：快速**

```text
1.0, 0.8571428571428571, 0.6, 0.0
```

**6 点 / 5 次 Euler 更新：平衡**
```text
1.0, 0.9230769230769231, 0.8181818181818182, 0.6666666666666666, 0.42857142857142855, 0.0
```

此配置兼顾速度与画质，适合在 ComfyUI 转换模型和量化版本上进行对比测试。

**8 点 / 7 次 Euler 更新：画质优先（可能过度锐化）**

```text
1.0, 0.9473684210526315, 0.8823529411764706, 0.8, 0.6923076923076923, 0.5454545454545454, 0.3333333333333333, 0.0
```

保留 **Euler**、**BasicGuider / CFG 1.0** 和视频/音频 shift **3.0 / 3.0**。末尾的 `0.0` 必须保留：它是最后一次更新的终点，无需在零处再执行模型。不要追加第二个零，也不要再次 shift 这些列表。选择 **4 点追求速度、6 点兼顾速度与画质、8 点偏重画质（可能过度锐化）**。这些是 ComfyUI 的实用预设选择，实际效果取决于视频和模型/量化版本。编码和最终解码的耗时不会随采样步数一起消失。

## 长视频生成

示例工作流（分窗口条件 + 分块采样器）：[example_workflows/viggle-animate-h3_workflow-chunked-sampler.json](example_workflows/viggle-animate-h3_workflow-chunked-sampler.json)。

将 **Viggle-Animate Conditioning (H3, Windowed)** 的 `cond_set` 接到 **Viggle Chunked Sampler**，`guider_positive` 接到 BasicGuider 的条件输入（或 CFGGuider 的 positive）。建议使用 **124 帧一块、`five_frame_anchor` 模式**。正常窗口前进 119 帧；5 帧经解码、重新编码后作为两个固定潜变量。`overlap_frames` 仅用于 `latent_overlap` 模式。参考图尽量使用驱动视频中某一帧的重绘版本，输入和输出都使用 24 fps。

使用标准 ComfyUI H3 VAE 时，分窗口条件节点会复用上一窗口中已编码的完整 17 帧块；每个窗口需要补帧的尾部仍单独编码。这样可以减少重复的 VAE 计算，无需降低分辨率、改变精度或增大编码窗口；自定义 VAE 包装类仍采用完整窗口编码。日志会显示复用的块数。分辨率越高、视频越长，编码仍然越耗时。

### 分块种子控制

| 参数 | 含义 |
|---|---|
| `seed` | 基础采样种子：第 1 块使用 `seed`，第 2 块使用 `seed + 1`，依此类推。两个 Viggle 采样器都在内部生成标准噪声，更新旧工作流时请移除 noise 连线。 |
| `rerender_chunk` | 要覆盖种子的分块编号，**从 1 开始**；**0 表示关闭覆盖**。请使用 `chunk_map` 中实际存在的编号。 |
| `rerender_seed` | 仅用于所选分块的替代种子。`rerender_chunk = 0` 时无效；种子数值 0 本身是有效的。 |

这些参数用于从某个效果不理想的分块开始尝试另一种结果，无需修改整段视频的基础种子。例如基础 `seed = 58` 时，四块的种子为 `58, 59, 60, 61`。设置 `rerender_chunk = 2`、`rerender_seed = 123` 后变为 `58, 123, 60, 61`。第 1 块可从缓存复用；覆盖值改变后，第 2–4 块需要重新生成，因为每块都依赖上一块传入的内容。后续块即使种子数值相同，结果也会受变化的重叠内容影响。

保持覆盖值即可保留这次选择；更换 `rerender_seed` 可尝试另一个结果。它是种子覆盖功能，并非强制刷新按钮：设置不变时仍可能命中缓存。将 `rerender_chunk` 改回 0 会恢复基础种子序列。`chunk_map` 会显示帧范围、种子、重叠量，以及 `[cached]`（缓存）或 `[rendered]`（本次生成）标记。

### 分块与重渲染的局限

- 不能只修改一块并保持后续所有块不变，因为重叠内容会向后传递。所选分块从上一块继承的开头仍被固定，不会随该块重渲染而重绘。
- 缓存仅在内存中，不是保存到磁盘的检查点或断点续跑功能。重启 ComfyUI 会清空缓存；缓存淘汰、输入/模型/采样设置改变或条目过大，都可能导致前面的块也重新生成。
- 标准 guider 对象、可检查的模型/采样器设置支持缓存复用。无法可靠检查的自定义选项、回调或补丁会跳过缓存，但仍正常采样；使用补丁的工作流可能每次重渲染全部分块。
- 分块缓存保留输出精度，CPU 张量存储上限为 **2 GiB**。参考编码另有 **256 MiB / 64 条**限制。超出容量的条目不会缓存。
- 即使前面的块命中缓存，最后仍会重新解码完整视频。整段条件、主潜变量、最终解码和输出帧仍占用内存；分块不能保证任意长度的视频都能放入内存/显存。
- 接缝处仍可能出现动作、身份或光照变化；重叠不保证完全无缝或无卡顿。`chunk_frames` 是窗口长度上限，最后一块向前移动以保持完整长度。所有已加载的参考帧都会使用；非网格长度向上生成到下一个 `17k+5` 边界，最多多生成 **16 帧**（最少 5 帧）。输出由模型采样，不是在末尾复制帧。

`latent_overlap` 模式下，362 帧输入、124 帧窗口和 22 帧重叠时，窗口为 **0–123、102–225、204–327、238–361**，
每块均为 **124 帧**。361 帧输入也以 362 帧为生成目标，不再截到 345 帧。
末尾参考在 VAE 编码前重复最后一帧以填满生成网格；Chunked Sampler 解码后裁回源帧数，高级循环的外部解码仍包含填充。
帧数以加载器实际输出的图像为准；帧率转换后可能与源视频元数据不同。
- 丢弃模型生成的音频。需要声音时，将驱动视频的音频接到视频保存节点，并与保留下来的视频长度对齐；输入和输出保持 **24 fps**。
- 无效 sigma 调度或含 NaN/Inf 的分块潜变量会触发明确错误，防止损坏结果进入缓存或传给后续分块。

### 分块循环节点

每个节点的完整说明（接口、输出槽顺序、恢复规则、典型流程）：[docs/long_video_guide.md](docs/long_video_guide.md)。实测示例工作流：[example_workflows/viggle-animate-h3_workflow-long-video-advanced.json](example_workflows/viggle-animate-h3_workflow-long-video-advanced.json)。

四个节点把同样的分窗口条件变成**图展开循环**，并配合磁盘检查点：每生成一块，就通过你自己的节点解码并保存 —— 锚点模式需将 H3 VAE 接到 Sample Chunk；中途失败时，已完成的块全部保留：

| 节点 | 作用 |
|---|---|
| **Viggle Chunk Loop Start** | 从 `cond_set` 读取分块计划，确定检查点目录，初始化循环状态。 |
| **Viggle Sample Chunk** | 只采样当前窗口。输出该块的视频 LATENT（供普通 VAE Decode 使用）以及携带状态；返回前先把潜变量写入磁盘检查点。 |
| **Viggle Chunk Loop End** | 等待本次迭代的解码/保存分支完成，然后展开下一块，或返回完成的集合。 |
| **Viggle Assemble Chunk Latents** | 把保存的分块拼接（去除重叠）为一个 LATENT，做一次最终解码。`chunk_number > 0` 时只加载某一块用于检查；中断的运行可部分拼接。 |

```text
Loop Start ─ loop ───────────────────────────────┐
     └ state → Sample Chunk → LATENT → VAE Decode ─┬→ Loop End (images)
                                                   └→ Video Combine → filenames ↗ (after_save)
```

- **解码/保存分支必须接回 Loop End** —— 把 VAE Decode 的 images 接到 `images`，Video Combine 的 `filenames` 输出接到 `after_save`，确保上一块完成解码保存后才开始下一块。（核心 SaveWEBM 也可以：其 `images` 输出直接接 `images`。）
- 检查点以 safetensors 加 `manifest.json` 的形式存放在 `output/viggle_chunks/<run_name>/`；写入是原子操作，崩溃不会留下半有效的块。
- 打开 `resume` 后重新排队，会恢复所有**图、模型、条件、sigma 和分块种子**仍匹配的块（检查点文件名内嵌该指纹）。设置改变会以新文件名重新采样；旧结果保留在磁盘上。`rerender_chunk` / `rerender_seed` 与单遍采样器一致。
- Long Video Advanced 示例中，测试恢复前请把 Sample Chunk 的种子控制从 `randomize` 改为 **`fixed`**。分块和最终 Video Combine 节点都要启用 **`save_output`** 才会永久保存视频；示例默认使用临时预览。
- 逐块解码的预览包含重叠上下文；最终成片请走 **Viggle Assemble Chunk Latents** → 一次 VAE Decode。
- 解码或保存预览失败时，当前块的潜变量检查点也已保存。使用相同 `run_name` 并启用 `resume`，即可重试预览，无需重新采样匹配的块。重放预览失败不会截短已有的完整 manifest。
- Sample Chunk 根据种子在内部生成标准噪声，不再提供 noise 输入；更新旧工作流时请移除旧 noise 连线。通过连线提供模型文件名时，会保守检查该模型类别下的文件元数据，因此同类别其他文件的变化也可能使恢复失效。代码更新会使自动恢复失效，但旧潜变量文件仍可读取。
- Sample Chunk 的只读 `live_progress` 文本框会实时显示每块的采样、恢复和解码/保存进度，无需额外节点或连线。更新后请重启 ComfyUI 并刷新浏览器。“循环完成”不代表下游最终拼接、解码和保存已完成；Start、End 和 Assemble 保留 STRING 状态输出。
- 每块解码及保存完成后，才开始采样下一块。ComfyUI 执行缓存可能在内存中保留各块图像（124 帧、1024×576、float32 RGB 每块约 0.82 GiB）。要永久保存预览视频，请打开 Video Combine 的 `save_output`；潜变量检查点独立保存。

## 已知局限

* **重新入镜时身份漂移**：当主体离开镜头后重新入镜时，重新出现的主体可能逐渐趋向驱动视频中的原始外观，而不是参考图中的角色。
* **大幅动作时身份保持下降**：当主体姿态与参考图差异很大，或者进行突然、剧烈的动作（例如后空翻）时，身份保持能力会减弱。与参考图姿态差异越大，reference identity 的约束通常越弱。
* **口型同步限制**：生成角色不会稳定地与驱动视频中的口型保持同步。
* **参考图兼容性**：如果输出中的角色无法很好地保持参考图身份，建议让参考图中的人物姿态 / 站姿尽可能接近驱动视频中的人物，并尽可能保持相似背景。遇到明显 identity drift 时，建议将生成分辨率控制在 **0.4–0.6 MP**，并尝试 **LCM 或 normal + 6–8 步**。更高分辨率实测到 **1.2 MP** 仍然可靠，但前提是驱动视频和参考图本身足够清晰、没有像素化。

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

Viggle Chunked Sampler 也提供 live_progress，显示逐块采样、缓存复用和最终解码。保留 chunk_map 详细报告；该采样器仅支持内存缓存。

## 问题反馈

* Issues：[ComfyUI-Viggle-Animate-H3/issues](https://github.com/Saganaki22/ComfyUI-Viggle-Animate-H3/issues)
