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

## 1.4.0 更新

新增实验性的分窗口条件节点和 **Viggle Chunked Sampler**，支持长视频分块生成、潜变量重叠传递、分块缓存复用，以及通过种子覆盖重试某一段。新增从上游 shift-3 调度推导的 **4–8 步自定义 sigma 预设**（按上游计数，即 4、6 或 8 个 sigma 点）。请根据用途和速度需求选择；较长的调度属于实验性扩展，不保证画质更好。

## 节点

| 节点 | 功能 |
|---|---|
| **Load Text Conditioning (Viggle)** | 从 `models/text_cond/` 下拉加载冻结文本条件 |
| **Viggle-Animate Conditioning (H3)** | 构建条件 + AV latent:视频优先的参考顺序,两个参考均按驱动视频短边嵌套 —— 即微调训练时使用的布局 |
| **Viggle-Animate Conditioning (H3, Windowed)** | 将驱动视频划分为重叠窗口，并为每块构建参考条件；输出 `cond_set` 接分块采样器，`guider_positive` 接 guider |
| **Viggle Chunked Sampler** | 逐块采样并保留上一块的重叠内容，复用符合条件的缓存，最后统一解码；输出视频帧 `frames` 和分块信息 `chunk_map` |

模型加载和采样控制使用 ComfyUI 核心节点：**Load Diffusion Model**、**Load LoRA (Model Only)**、
**ModelSamplingMiniMaxH3**（视频/音频 shift 均为 3.0）、**BasicGuider**、**KSamplerSelect** 和 **ManualSigmas**。
也可使用 KJNodes 的 **CustomSigmas** 输入下方调度。分块采样器内部已完成解码，`frames` 直接连接视频保存节点。

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

- **上游采样基准：** 使用 **ManualSigmas** 输入 `1.0, 0.8571428571428571, 0.6, 0.0`，搭配 **Euler**、**BasicGuider**（或 CFG 1.0），以及原始 `viggle_animate_dmd_lora.safetensors`，强度 1.0。将 sigma 输出连接到 SamplerCustomAdvanced 或 Viggle Chunked Sampler。四个 sigma 点对应 **3 次模型计算**，即上游所称的“4 步”。
- **ModelSamplingMiniMaxH3** 的视频、音频 shift 都保留 **3.0**。它不会再次变换手动输入的 sigma 列表；不要在列表后再接 sigma 变换节点。
- **KJNodes CustomSigmas：** 上述四个值应配合 `interpolate_to_steps = 3`。设为 4 会对包含零的序列做对数插值，得到以 `0, 0` 结尾的调度；Euler 随后除以零，产生 NaN，导致最终视频和后续分块变黑。分块采样器现在会在渲染前拒绝这种无效调度。
- **Load Video：** 使用 `force_rate = 24`。单段生成时，`frame_load_cap` 与条件节点的 `length` 一致（例如 124）。分窗口生成时加载所需的完整视频；VHS 的 `frame_load_cap = 0` 表示加载全部帧。
- **width/height 设为 0** 表示继承驱动视频尺寸；显式设置时决定输出画布，每轴取整到 32。测试画布范围：**0.4–0.98 百万像素**。
- 其他采样器、调度和 rank-64 LoRA 属于可尝试的替代方案。旧版示例工作流使用的 LCM / bong_tangent 八步配置与上游基准不同。
- 可叠加 Comfy Kitchen 和 block sparse attention 补丁。

## 自定义 sigma 预设（上游计数 4–8 步）

将下方任一列表粘贴到 **ManualSigmas** 或 KJNodes **CustomSigmas**，并将其 `SIGMAS` 输出连接采样器。公式为 `sigma = 3*t / (1 + 2*t)`，其中 `t` 从 1 到 0 等间隔取值。四点预设匹配上游基准，六点和八点预设按同一规律扩展。

| 建议的工作流节点标题 | sigma 点数（包含末尾零） | 采样更新次数 / KJNodes `interpolate_to_steps` |
|---|---|---|
| **Viggle DMD — 3 Steps (Upstream “4-Step”)** | 4 | **3** |
| **Viggle — 5 Steps (6 Sigma Points)** | 6 | **5** |
| **Viggle — 7 Steps (8 Sigma Points)** | 8 | **7** |

**4 点：最快的基准配置**

```text
1.0, 0.8571428571428571, 0.6, 0.0
```

**6 点：中等采样开销**

```text
1.0, 0.9230769230769231, 0.8181818181818182, 0.6666666666666666, 0.42857142857142855, 0.0
```

**8 点：更多采样更新**

```text
1.0, 0.9473684210526315, 0.8823529411764706, 0.8, 0.6923076923076923, 0.5454545454545454, 0.3333333333333333, 0.0
```

保留 **Euler**、**BasicGuider / CFG 1.0** 和视频/音频 shift **3.0 / 3.0**。末尾的 `0.0` 必须保留：它是最后一次更新的终点，无需在零处再执行模型。不要追加第二个零，也不要再次 shift 这些列表。减少更新次数可提高速度；较长调度是否改善效果应在自己的素材上比较。编码和最终解码的耗时不会随采样步数一起消失。

## 实验性长视频（`exp`）

将 **Viggle-Animate Conditioning (H3, Windowed)** 的 `cond_set` 接到 **Viggle Chunked Sampler**，`guider_positive` 接到 BasicGuider 的条件输入（或 CFGGuider 的 positive）。建议从 **124 帧一块、22 帧重叠**开始。重叠区域保留上一块的输出，完整潜变量拼好后统一解码。参考图尽量使用驱动视频中某一帧的重绘版本，输入和输出都使用 24 fps。

使用标准 ComfyUI H3 VAE 时，分窗口条件节点会复用上一窗口中已编码的完整 17 帧块；每个窗口需要补帧的尾部仍单独编码。这样可以减少重复的 VAE 计算，无需降低分辨率、改变精度或增大编码窗口；自定义 VAE 包装类仍采用完整窗口编码。日志会显示复用的块数。分辨率越高、视频越长，编码仍然越耗时。

### 分块种子控制

| 参数 | 含义 |
|---|---|
| `seed` | 基础采样种子：第 1 块使用 `seed`，第 2 块使用 `seed + 1`，依此类推。分块采样器以此覆盖连接的噪声节点中的种子。 |
| `rerender_chunk` | 要覆盖种子的分块编号，**从 1 开始**；**0 表示关闭覆盖**。请使用 `chunk_map` 中实际存在的编号。 |
| `rerender_seed` | 仅用于所选分块的替代种子。`rerender_chunk = 0` 时无效；种子数值 0 本身是有效的。 |

这些参数用于从某个效果不理想的分块开始尝试另一种结果，无需修改整段视频的基础种子。例如基础 `seed = 58` 时，四块的种子为 `58, 59, 60, 61`。设置 `rerender_chunk = 2`、`rerender_seed = 123` 后变为 `58, 123, 60, 61`。第 1 块可从缓存复用；覆盖值改变后，第 2–4 块需要重新生成，因为每块都依赖上一块传入的内容。后续块即使种子数值相同，结果也会受变化的重叠内容影响。

保持覆盖值即可保留这次选择；更换 `rerender_seed` 可尝试另一个结果。它是种子覆盖功能，并非强制刷新按钮：设置不变时仍可能命中缓存。将 `rerender_chunk` 改回 0 会恢复基础种子序列。`chunk_map` 会显示帧范围、种子、重叠量，以及 `[cached]`（缓存）或 `[rendered]`（本次生成）标记。

### 分块与重渲染的局限

- 不能只修改一块并保持后续所有块不变，因为重叠内容会向后传递。所选分块从上一块继承的开头仍被固定，不会随该块重渲染而重绘。
- 缓存仅在内存中，不是保存到磁盘的检查点或断点续跑功能。重启 ComfyUI 会清空缓存；缓存淘汰、输入/模型/采样设置改变或条目过大，都可能导致前面的块也重新生成。
- 标准噪声和 guider 对象、可检查的模型/采样器设置支持缓存复用。无法可靠检查的自定义选项、回调或补丁会跳过缓存，但仍正常采样；使用补丁的工作流可能每次重渲染全部分块。
- 分块缓存保留输出精度，CPU 张量存储上限为 **2 GiB**。参考编码另有 **256 MiB / 64 条**限制。超出容量的条目不会缓存。
- 即使前面的块命中缓存，最后仍会重新解码完整视频。整段条件、主潜变量、最终解码和输出帧仍占用内存；分块不能保证任意长度的视频都能放入内存/显存。
- 接缝处仍可能出现动作、身份或光照变化；重叠不保证完全无缝或无卡顿。最后一个窗口可能有更多重叠；为适配 `17k+5` 帧网格，最多会丢弃末尾 **16 帧**。
- 丢弃模型生成的音频。需要声音时，将驱动视频的音频接到视频保存节点，并与保留下来的视频长度对齐；输入和输出保持 **24 fps**。
- 无效 sigma 调度或含 NaN/Inf 的分块潜变量会触发明确错误，防止损坏结果进入缓存或传给后续分块。

### 分块循环节点（实验性）

每个节点的完整说明（接口、输出槽顺序、恢复规则、典型流程）：[docs/experimental_nodes.md](docs/experimental_nodes.md)。实测示例工作流：[example_workflows/viggle-animate-h3_workflow_chunked_window_exp.json](example_workflows/viggle-animate-h3_workflow_chunked_window_exp.json)。

四个节点把同样的分窗口条件变成**图展开循环**，并配合磁盘检查点：每生成一块，就通过你自己的节点解码并保存 —— 采样器不再需要 VAE 输入；中途失败时，已完成的块全部保留：

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
- 逐块解码的预览包含重叠上下文；最终成片请走 **Viggle Assemble Chunk Latents** → 一次 VAE Decode。
- 中途终止的运行会在磁盘上保留第 1…k−1 块，manifest 仍然有效：用相同 `run_name` 重新排队即可，只补采缺失的块。

## 已知局限

- **重新入镜时身份漂移**:当主体离开镜头后重新入镜,画面会趋向驱动视频中的原始外观,
  而非参考图。当主体大幅偏离参考姿态或做出剧烈动作(如后空翻)时同样如此 ——
  与参考图的姿态差距越大,身份保持越弱。
- **口型同步：** 生成角色不能可靠地保持与驱动视频一致的口型同步。
- **参考图兼容性：** 身份保持较差时，让参考图人物的姿态、站位和背景尽量接近驱动视频；先在 **0.4–0.6 百万像素**下测试，再比较不同采样设置。

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
