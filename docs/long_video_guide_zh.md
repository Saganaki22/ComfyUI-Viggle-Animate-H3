# 长视频生成指南

[中文 README](../README_zh.md) · [English / 完整插槽表](long_video_guide.md)

更新后重启 ComfyUI 并刷新浏览器，加载实时进度扩展。

## 选择工作流

| 用途 | 工作流 | 恢复方式 |
|---|---|---|
| 单镜头，通常 124 帧（24 fps 下约 5.2 秒） | 原始 Conditioning → 核心采样器 → VAE Decode → 保存 | 普通工作流 |
| 长片段，接线简单，只在最后解码 | Windowed Conditioning → Chunked Sampler | 仅内存缓存，重启后消失 |
| 每块预览、单独保存，取消/重启后继续 | Windowed Conditioning → Start / Sample Chunk / End → Assemble → VAE Decode | 磁盘检查点 |

[循环示例](../example_workflows/viggle-animate-h3_workflow-chunked-window-advanced.json) ·
[单遍分块示例](../example_workflows/viggle-animate-h3_workflow-chunked-sampler.json)

循环只需要放置一个 Sample Chunk。Windowed Conditioning 自动决定块数，End 自动重复执行，
不需要按视频长度手工增加采样节点。即使只有两块，需要保留进度时也适合使用循环。

## 节点和容易混淆的插槽

| 节点 | 作用与接线 |
|---|---|
| Load Text Conditioning | 加载固定文本条件，接条件节点的 `text_cond` |
| 原始 Conditioning (H3) | 单段生成的条件和 AV 潜变量，接普通采样流程 |
| Windowed Conditioning | 分配重叠窗口并编码参考；`cond_set` 接 Start 或 Chunked Sampler，`guider_positive` 接 guider |
| Chunked Sampler | 一个节点内采样所有块并最终解码；输出 `frames` 直接保存、`chunk_map` 查看种子和范围；内部需要 VAE |
| Loop Start | 选择目录并初始化循环；`state` 接 Sample Chunk，`loop` 接 End；`initial_state` 留空，循环内部自动传入上一轮状态 |
| Sample Chunk | 采样当前块并先保存潜变量；`chunk` 接 End，`video_latent` 接外部 VAE Decode，没有 VAE 输入 |
| Loop End | `loop` 接 Start，`chunk` 接 Sample Chunk，`images` 接解码图像；`after_save` 可接 Video Combine 的 `filenames`，确保保存结束才继续 |
| Assemble Chunk Latents | `chunks` 接 End；`chunk_number = 0` 拼接全部，正数加载单块（包含重叠）；输出 LATENT 供最终解码 |

Sample Chunk 的 `filename_prefix` 是保存路径前缀，包含运行名、块编号和指纹，
可连接保存节点的文件名前缀输入。它不是状态文字。

## 首次设置

1. 加载模型、DMD LoRA、H3 视频 VAE 和固定文本条件，文件位置见 README。
   循环示例使用 VHS 视频节点，需要 VideoHelperSuite；按界面提示补齐其他缺失节点。
2. 驱动视频设为 `force_rate = 24`；VHS 的 `frame_load_cap = 0` 加载全部帧。
   从 `chunk_frames = 124`、`overlap_frames = 22` 开始，参考图尽量匹配驱动镜头的姿态和构图。
3. 接好上表中的 Start / Sample Chunk / End。Sample Chunk 的 LATENT 接 VAE Decode，
   解码图像同时接 End `images` 和 Video Combine；其 `filenames` 接 End `after_save`。
4. 将 Sample Chunk `filename_prefix` 接 Video Combine 文件名前缀，必要时先把该控件转为输入。
   **要永久保存视频，请启用 `save_output`**；示例默认预览是临时文件，潜变量检查点独立保存。
5. End `chunks` 接 Assemble，`chunk_number = 0`，再接最终 VAE Decode 和保存节点。
   输出也设为 24 fps。最终视频使用驱动音频并裁到保留的视频长度；不能直接拼接带重叠的预览视频。
6. Start 填一个新的 `run_name`，开启 `resume`；Sample Chunk 使用固定 `seed`，
   `rerender_chunk = 0`。采样配置见 README 的 sigma 预设：四点基准配 Euler、CFG 1.0，
   ModelSamplingMiniMaxH3 视频/音频 shift 都为 3.0；不要再变换 sigma 列表。

四点列表 `1.0, 0.8571428571428571, 0.6, 0.0` 对应三次采样更新。
KJNodes CustomSigmas 的 `interpolate_to_steps` 设为 3；六点/八点预设分别设为 5/7。
末尾保留一个零。六点、八点是同规律的实验扩展，不保证画质提高。

## 目录、恢复与重渲染

Start 自动创建 `ComfyUI/output/viggle_chunks/<run_name>/`，保存 safetensors 潜变量和
`manifest.json`。新名字对应独立目录；相同名字加 `resume = true` 恢复匹配的块。
`resume = false` 会重新采样。不要让并发任务写入同一个运行目录。

检查点在外部解码之前写入：解码或预览保存失败时，当前块通常无需重新采样。
重放匹配预览失败不会截短已有完整 manifest。恢复会再次解码/保存预览，并非跳过全部计算。

`seed` 是基础种子，第 N 块使用 `seed + N − 1`。
`rerender_chunk` 从 1 开始计数，0 关闭覆盖；`rerender_seed` 只替换指定块的种子。
改变第 k 块后，k 及后续块因传递内容改变而重新采样，前面的匹配块可恢复。
该块继承的重叠仍被固定，不会整体重画。相同覆盖配置可恢复相同结果。

恢复检查图、模型文件元数据、条件、sigma、窗口和有效种子。
Sample Chunk 根据有效分块种子在内部创建标准随机噪声，不再需要外部 noise 输入。更新旧工作流时请移除旧 noise 连线。
连线提供模型文件名时会保守检查整个模型类别的文件元数据，其他文件变化也可能使恢复失效。
代码更新会改变指纹，旧检查点仍可读取，但不会自动作为新代码的采样结果复用。

只想读取已有结果：断开 Assemble `chunks`，填写已有 `run_name`；0 加载 manifest 中全部块，
正数加载指定单块。Assemble 不创建目录。中断的运行可以输出部分视频，缺失或损坏文件会报错。

## 状态显示在哪里？

| 显示 | 含义 |
|---|---|
| Sample Chunk `live_progress` | 同一节点实时更新：采样/恢复 → 检查点已保存/恢复 → 解码/保存 → 循环完成 |
| Start `status` | 当前迭代的块编号，普通 STRING |
| End `status` | 循环完成及目录；下游最终解码可能仍在执行 |
| Assemble `status` | 完整/部分、块数及帧数，或指定块信息 |

STRING 可接 Show Text，但循环展开后可能只显示某次迭代；实时查看请用 `live_progress`。
它无需连线，只向提交任务的浏览器发送更新，刷新后不会恢复历史进度。
前置条件/VAE 编码尚未完成时可能显示 Waiting；这不表示流程停止。

## 两块恢复测试

使用约 10 秒视频和默认窗口设置（按帧网格裁剪后通常两块）。
在第二块采样中取消，保持设置不变、启用 `resume` 重排队。
第一块应恢复并重新解码；第二块重新采样，除非取消前已经保存检查点。
完成后再次排队应全部恢复。再将 `rerender_chunk = 2` 并更换 `rerender_seed`，验证只重采后缀。

## 长视频限制

- 视频按 `17k+5` 帧网格对齐，最多丢弃末尾 16 帧；最后一块可能有更大重叠，短片段窗口可能更短。
- 重叠传递不保证无卡顿、无身份漂移；剧烈动作、出入镜、镜头切换仍可能变形，适合按镜头分开生成。
- 原始视频、全部条件、最终潜变量和最终解码仍需要内存。ComfyUI 缓存可能保留每块解码图像；
  124 帧、1024×576、float32 RGB 每块约 0.82 GiB。循环不是恒定内存方案。
- 标准 H3 VAE 可复用完整的重叠 17 帧编码块；新块和填充尾部仍需编码，自定义 VAE 使用完整窗口路径。
  分辨率和片长越大，前置编码仍越慢。
- 潜变量检查点与预览视频是两套文件；旧重渲染文件会累积，确认不需要后自行清理运行目录。
