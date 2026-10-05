# CloudStudio × MiniMax-H3 Ref2VA 应用模板（A10 量化版）

面向 [CloudStudio（腾讯云云端 IDE）](https://cloudstudio.net/) 的 [MiniMax-H3](https://github.com/MiniMax-AI/MiniMax-H3) **Ref2VA（参考输入生音视频）** 应用模板。默认方案针对 **单张 NVIDIA A10（24GB）** 设计：使用 **ConvRot INT8 + NVFP4-AWQ 量化权重（下载约 43GB）+ layerwise CPU/NVMe offload**，依据 SGLang 官方「1×RTX 4090 24GB」实测配方（A10 与 RTX 3090/4090 同为 24GB 档位，计算能力 SM8.6，INT8 内核走 comfy-kitchen）。

提供四个「一键」能力：

1. **一键初始化** —— `scripts/setup.sh` 完成 CUDA、ffmpeg、SGLang[diffusion]、comfy-kitchen 安装与量化权重下载；
2. **CUDA 一键安装** —— 自动识别驱动版本安装 CUDA Toolkit，并安装必需的 ffmpeg/ffprobe；
3. **量化模型一键下载** —— 从魔搭下载 Comfy-Org 量化权重（公开可访问，无需申请）+ 官方精简配置树，断点续传；
4. **一键启动 / 生成** —— `scripts/serve.sh` 自动识别量化权重并以单卡 offload 配方启动（端口 30011）；`scripts/request-ref2va.sh` 自动完成提交任务、轮询、下载 MP4。

## 为什么 A10 24GB 能跑

| 事实 | 说明 |
| --- | --- |
| 全量 BF16 权重 | Ref2VA 约 144GB（DiT 66.3GB + Qwen3-VL 文本编码器 61.7GB + VAE），需多卡数据中心级 GPU（官方多卡配方 4×H100/H200） |
| AdaLN 精简（pruned） | 33B 参数中约 13B 是 AdaLN 调制分支，推理时可预计算/缓存，**不需要加载**，DiT 从 66.3GB 降到 40.2GB（BF16） |
| ConvRot INT8 | 基于分组正则哈达玛旋转的即插即用低比特量化（ConvRot，[arXiv:2512.03673](https://arxiv.org/abs/2512.03673)），以 W8A8 INT8 应用于 H3 精简 DiT，仅 **20.97GB**；相对 BF16 有轻微精度损失（官方实测 PSNR 24.81dB）。SM8.6 上经 [comfy-kitchen](https://pypi.org/project/comfy-kitchen/) 内核执行（SGLang 自带 JIT 算子仅覆盖 Hopper/Blackwell） |
| NVFP4-AWQ 文本编码器 | **15.69GB**；属「压缩存储、BF16/FP16 计算」的内存型格式，**不需要 Blackwell**（Comfy-Org 官方说明） |
| layerwise offload | 权重不全驻留显存，逐层在 主机内存/NVMe → 显存 间调度；VAE 保持常驻 |
| 实测参考 | 官方在单张 RTX 4090 24GB 上完成 1344×768/107 帧/20 步 Ref2VA 同级负载，**显存峰值约 18GB**，INT8+FlashAttention 端到端约 303 秒（PSNR 24.81dB vs BF16）。A10 显存同为 24GB，可容纳同样的量化组合；但显存带宽 600 GB/s（4090 为 1008 GB/s，A10 约为其六成），INT8 算力也更低，生成耗时会明显长于该实测值。offload 场景下逐层权重传输走 PCIe Gen4，两者平台相当，实际差距以实测为准 |

## 下载内容（默认量化组合，合计约 42.5GB）

| 组件 | 文件（魔搭 `Comfy-Org/MiniMax-H3`） | 大小 |
| --- | --- | --- |
| DiT | `diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors` | 20.97 GB |
| 文本编码器 | `text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` | 15.69 GB |
| 视频 VAE | `vae/minimax_h3_video_vae_fp16.safetensors` | 5.21 GB |
| 音频 VAE | `vae/minimax_h3_audio_vae_fp32.safetensors` | 0.61 GB |

另外从官方 `MiniMax/MiniMax-H3` 下载几十 MB 的精简配置树（`model_index.json`、Ref2VA 的 processor/tokenizer/组件 config 与 VAE 代码，不含任何权重分片）。SGLang 通过 `--component-weights-paths.*` 用量化文件覆盖各组件权重，`--model-path` 仍指向配置树根目录。

> 不建议在 A10 上使用 `fp8_scaled` 量化文件——Ampere 架构无原生 FP8 支持。可选变体见下文「量化变体」。

## 关于 MiniMax-H3 Ref2VA

MiniMax-H3 是开源的通用多模态生成模型，DiT 联合生成 **24fps H.264 视频 + 原生立体声音频**（4–15 秒，768p 短边）。Ref2VA 接受参考输入生成新镜头：

| 项目 | 说明 |
| --- | --- |
| 参考输入 | 图像 ≤9 张；视频 ≤3 段（每段 2–15s，总计 ≤15s）；音频 ≤3 段且**必须与图像/视频搭配**；混合输入 ≤12 个文件 |
| 文件要求 | 视频 H.264/H.265 ≤50MB；图片 jpg/png/webp/heic ≤30MB；音频 wav/mp3 ≤15MB；整体请求 <64MB，大文件优先用 URL |
| 输出 | 单个 MP4：H.264 24fps + 立体声 AAC 32kHz |

## 使用流程

> CloudStudio 通过 **Git 仓库导入** 创建应用；环境配好后用「文件 → 发布自定义模板」生成团队模板与分享链接。

### 第一步：从 Git 仓库创建应用并选择算力

1. 将本仓库推送到 GitHub / Gitee / CNB；
2. 登录 [cloudstudio.net](https://cloudstudio.net/) → 「创建应用」→ **从 Git 仓库导入**（仓库克隆到 `/workspace`）；
3. 基础镜像选择 **Ubuntu + Python 3.10+**；
4. 算力选择 **NVIDIA A10（24GB）单卡**（或 RTX 3090/4090 等 24GB+ 卡）；
5. **主机内存建议 ≥64GB**（权重优先在内存中 pin 驻；32GB 也可运行，但主要靠 NVMe 流式加载，速度明显更慢）；**系统盘必须是 NVMe SSD**（SATA SSD 不可用），预留 **≥60GB** 磁盘。

### 第二步：一键初始化

```bash
bash /workspace/scripts/setup.sh
```

依次完成：安装 `.vscode/preview.yml` → CUDA Toolkit + ffmpeg → `sglang[diffusion]` + modelscope + **comfy-kitchen** → 下载量化权重（官方精简树到 `/workspace/models/MiniMax-H3`，量化文件到 `/workspace/models/Comfy-MiniMax-H3`）。全过程幂等，中断后重跑即可续传。

```bash
SETUP_SKIP_MODEL=1 bash /workspace/scripts/setup.sh   # 只装环境
H3_WEIGHTS=online bash /workspace/scripts/setup.sh    # 改下官方 BF16 全量（约144GB，供在线量化）
```

### 第三步：启动服务

- **方式一**：点击顶部「运行」，30011 端口出现在「端口」面板；
- **方式二**：终端执行 `bash /workspace/scripts/serve.sh`。

量化模式实际执行（`serve.sh` 自动检测并拼装，路径以本机为准）：

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
sglang serve \
  --model-path /workspace/models/MiniMax-H3 \
  --model-variant ref2va \
  --num-gpus 1 --ulysses-degree 1 \
  --component-weights-paths.transformer /workspace/models/Comfy-MiniMax-H3/diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors \
  --component-weights-paths.text_encoder /workspace/models/Comfy-MiniMax-H3/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors \
  --component-weights-paths.video_vae /workspace/models/Comfy-MiniMax-H3/vae/minimax_h3_video_vae_fp16.safetensors \
  --component-weights-paths.audio_vae /workspace/models/Comfy-MiniMax-H3/vae/minimax_h3_audio_vae_fp32.safetensors \
  --attention-backend fa \
  --performance-mode memory \
  --layerwise-offload-components dit,text_encoder \
  --dit-offload-prefetch-size 1 \
  --dit-layerwise-resident-layers 0 \
  --enable-torch-compile false \
  --host 0.0.0.0 --port 30011
```

依据：SGLang Cookbook「1×RTX 4090 24GB」配方。注意 **VAE 不放进 offload 列表**（避免 167 个解码 tile 各重复流式传输约 9GB）；预量化文件自描述，**不能再加 `--quantization`**。

### 第四步：发起 Ref2VA 生成

```bash
# 内置官方公开参考视频示例：提交 → 轮询 → 下载 MP4 到 /workspace/outputs/
bash /workspace/scripts/request-ref2va.sh

# 使用自己的参考图片 / 视频（URL 方式，服务端拉取）
REF_IMAGE_URL='https://example.com/ref.jpg' \
PROMPT='让照片中的人物微笑着挥手打招呼，保持场景与光线一致，并生成同步环境音' \
bash /workspace/scripts/request-ref2va.sh
```

直接调用 API（异步）：

```bash
# 1) 提交任务
curl -sS -X POST http://localhost:30011/v1/videos \
  -H 'Content-Type: application/json' -d '{
    "task": "ref2va",
    "prompt": "保持原视频的人物与场景，让人物挥手打招呼并生成同步环境音",
    "conditions": [
      {"type": "video", "uri": "https://example.com/ref.mp4", "role": "reference"}
    ],
    "target": {"short_edge": 768, "aspect_ratio": "auto", "duration_seconds": 5},
    "seed": 0
  }'

# 2) 查询状态：GET /v1/videos/{id}（queued / processing / completed / failed）
# 3) 下载成片：GET /v1/videos/{id}/content -o ref2va.mp4
```

### 第五步：发布为团队自定义模板（可选）

IDE 中点击 **「文件 → 发布自定义模板」**，填写信息后发布，复制分享链接或 Markdown 徽章即可；环境更新后可再次发布，链接不变。

## 量化变体（默认之外的选择）

下载时通过环境变量切换（需与启动时的权重文件对应，`serve.sh` 会自动识别目录中已下载的文件）：

```bash
# DiT 更省显存的 W6A8（约 16GB，画质损失更大；要求 comfy-kitchen>=0.2.27）
H3_DIT_QUANT=w6a8 bash /workspace/scripts/download-model.sh

# 不做 AdaLN 精简的 INT8（约 34GB，保真度更高）
H3_DIT_QUANT=int8_convrot bash /workspace/scripts/download-model.sh

# 文本编码器用 ConvRot INT8（约 27.1GB）替代 NVFP4-AWQ
H3_TE_QUANT=int8_convrot bash /workspace/scripts/download-model.sh
```

| 方案 | DiT | 文本编码器 | 适用 |
| --- | --- | --- | --- |
| **默认** | pruned int8_convrot（20.97GB） | nvfp4_awq（15.69GB） | A10 24GB 推荐 |
| 更省显存 | pruned w6a8（15.98GB） | nvfp4_awq（15.69GB） | 16GB 卡/显存吃紧时尝试 |
| 更高保真 | int8_convrot（34.04GB） | int8_convrot（27.14GB） | 主机内存充裕、可接受更慢 |
| 在线量化 | 官方 BF16 全量（144GB） | 同左 | 官方实测路径；`H3_WEIGHTS=online` 下载，单卡自动 `--quantization convrot_int8` |

## 性能调节

| 环境变量 | 默认 | 说明 |
| --- | --- | --- |
| `SGLANG_DIT_RESIDENT_LAYERS` | `0` | 常驻显存的 DiT 层数。主机内存/显存有余量时调大可显著加速（如 768p 下谨慎加到 4–6；显存接近上限时先调回 0） |
| `SGLANG_OFFLOAD_PREFETCH` | `1` | DiT 逐层预取深度 |
| `SGLANG_ATTENTION_BACKEND` | `fa` | 精确注意力；官方测过 `sol_attn`/`sage_attn` 更快但改变注意力数值、降低 PSNR，属有损选项 |
| `SGLANG_NUM_GPUS` | 量化 `1` / 全精度自动 | 量化主要收益是省显存：官方实测 convrot_int8 相对 BF16 单卡提速约 1.07×、4 卡 Ulysses 约 1.15×；多卡并行推荐 Ulysses 序列并行（`--ulysses-degree` = GPU 数） |

量化模式首次启动会编译/初始化内核，加载阶段较慢，属正常现象。生成耗时与主机内存、磁盘强相关：权重在主机内存中 pin 驻时最快；内存不足时每个去噪步从 NVMe 读取几十 GB，**务必使用 NVMe**。

## 工作目录与路径约定

| 内容 | 路径 |
| --- | --- |
| 仓库代码 / 脚本 | `/workspace`（脚本目录 `/workspace/scripts`） |
| Python 虚拟环境 | `/workspace/.venv`（新开终端自动激活） |
| 官方精简配置树 | `/workspace/models/MiniMax-H3`（`model_index.json` + `Ref2VA/`） |
| 量化权重 | `/workspace/models/Comfy-MiniMax-H3`（`diffusion_models/`、`text_encoders/`、`vae/`） |
| ModelScope 缓存 | `/workspace/.cache/modelscope`（`MODELSCOPE_CACHE`） |
| 生成结果 | `/workspace/outputs/` |
| 运行配置 | `/workspace/.vscode/preview.yml`（setup 自动安装） |

> `.venv/`、`models/`、`.cache/`、`outputs/`、`.vscode/` 均已写入 `.gitignore`。

## 目录结构

```
/workspace
├── config/
│   └── preview.yml            # CloudStudio 运行配置源（setup 复制到 .vscode/）
├── scripts/
│   ├── setup.sh               # 一键全量初始化（新空间只跑这一条）
│   ├── install-cuda.sh        # CUDA Toolkit + ffmpeg 一键安装
│   ├── install-sglang.sh      # SGLang[diffusion] + modelscope + comfy-kitchen
│   ├── download-model.sh      # 量化权重（默认）/ 官方全量 下载
│   ├── serve.sh               # Ref2VA 服务启动（自动识别量化配方）
│   ├── request-ref2va.sh      # Ref2VA 生成：提交/轮询/下载
│   └── fix-comfy-int8-embedding.sh  # 修复 SGLang「量化TE + layerwise offload」设备不一致 bug
├── .vscode/preview.yml        # 运行配置（setup 自动生成，已被 git 忽略）
├── models/ .cache/ outputs/ .venv/   # 运行时目录（git 已忽略）
├── .gitattributes / .gitignore
└── README.md
```

## 分步命令

```bash
bash /workspace/scripts/install-cuda.sh                 # CUDA + ffmpeg（已装/无 GPU 自动跳过）
bash /workspace/scripts/install-sglang.sh               # sglang[diffusion] + modelscope + comfy-kitchen
bash /workspace/scripts/download-model.sh               # 默认量化组合（约43GB）
H3_WEIGHTS=online bash /workspace/scripts/download-model.sh  # 官方 BF16 全量（约144GB）
MAX_WORKERS=8 bash /workspace/scripts/download-model.sh # 调整下载并发
bash /workspace/scripts/serve.sh                        # 自动选配方启动（端口 30011）
bash /workspace/scripts/request-ref2va.sh               # 生成并下载 MP4
```

## 常见问题

**Q：A10 没有 FP8，为什么量化方案能用？**
默认量化是 **ConvRot INT8（W8A8 整数）**，不是 FP8；文本编码器的 NVFP4-AWQ 是「压缩存储、BF16 计算」的内存型格式，两者都不依赖 Blackwell/Hopper 的 FP8/NVFP4 硬件。ConvRot INT8 有 jit 与 comfy-kitchen 两个内核后端，SGLang 加载时逐层自动选择，A10（SM8.6）上走 `comfy-kitchen`；也可用 `SGLANG_DIFFUSION_CONVROT_INT8_BACKEND` 强制指定。切勿使用 `fp8_scaled` 文件。

**Q：和 ComfyUI 是什么关系？直接用 ComfyUI 不行吗？**
权重文件来自 Comfy-Org 的重新打包（魔搭镜像），但推理引擎仍是 **SGLang 原生 H3 管线**（`/v1/videos` HTTP API），不是 ComfyUI。SGLang 官方明确支持加载 Comfy 单文件格式（`--component-weights-paths.*`，格式自动探测）。

**Q：启动时报 `unsupported GNU version! gcc versions later than 12 are not supported`（JIT 编译失败）？**
基础镜像的 gcc 比预装 nvcc 支持的上限新（如 CUDA 12.2 只支持 gcc≤12，而 Ubuntu 24.04 自带 gcc13），SGLang 启动时需要用 nvcc 现场编译 QKNorm+RoPE 等 JIT 内核。两种解决方式：

**方式 A（推荐，约 100MB）**：安装旧版编译器，通过 `NVCC_CCBIN`（等效 nvcc 的 `-ccbin`，JIT 编译子进程自动继承）指定：

```bash
apt-get update && apt-get install -y gcc-12 g++-12
# 之后每次启动都带该前缀（CloudStudio 内为 root；非 root 环境前面加 sudo）：
NVCC_CCBIN=/usr/bin/g++-12 bash scripts/serve.sh
# 也可写入 shell 配置免去每次手打：echo 'export NVCC_CCBIN=/usr/bin/g++-12' >> ~/.bashrc
```

**方式 B**：升级 CUDA Toolkit（3–5GB 下载，见下一问），升级后系统 gcc 直接被新 nvcc 接受。

**Q：可以升级 CUDA 吗？**
可以，但**不建议仅为解决 JIT 编译问题而升级**：pip 安装的 PyTorch 自带整套 CUDA 运行时，系统 nvcc 只用于 JIT 内核编译，升级 CUDA 对推理性能/显存/速度没有提升（GCC 版本门槛用上面的方式 A 约 100MB 即可解决）。确有需要（如以后要用新工具链特性）时按下面操作：

```bash
nvidia-smi | head -1                        # 先确认驱动：Driver ≥ 525.60.13 才兼容 CUDA 12.x 工具链
CUDA_FORCE=1 bash scripts/install-cuda.sh   # 强制按驱动能力重装/升级 CUDA Toolkit（NVIDIA 官方 apt 仓库，3-5GB）
source ~/.bashrc && nvcc --version          # 确认新版本生效；/usr/local/cuda 链接由安装包自动切换
bash scripts/serve.sh                       # 新 nvcc 支持系统 gcc13，无需 NVCC_CCBIN 直接启动
```

升级后首次启动会重新编译全部 JIT 内核（旧缓存按 nvcc 版本分目录存放，互不干扰），耗时稍长属正常。

**Q：启动时报找不到量化内核 / convrot？**
确认安装了 comfy-kitchen：`source /workspace/.venv/bin/activate && pip show comfy-kitchen`；没有则重跑 `bash /workspace/scripts/install-sglang.sh`。使用 W6A8 变体需 `comfy-kitchen>=0.2.27`。

**Q：首次生成报 `Expected all tensors to be on the same device, but got index is on cpu, different from other tensors on cuda:0`（`comfy_int8.py` 第 93 行）？**
SGLang 的 bug，仅「量化文本编码器 + `--layerwise-offload-components text_encoder`」组合触发：TE 词表（≥256MiB）走「驻留主机内存」优化，`weight` 被搬到 CPU、token 索引也被 hook 强制送 CPU 做 gather，但 INT8 量化层反量化用的 `weight_scale` 仍留在显存，第二级查表设备不一致即崩（bf16 全精度无 `weight_scale`，不受影响）。修复：

```bash
bash /workspace/scripts/fix-comfy-int8-embedding.sh   # 幂等，自动备份为 *.bak-pre-int8fix
bash /workspace/scripts/serve.sh                      # 重启服务生效
```

**Q：显存不足（OOM）？**
确认 `SGLANG_DIT_RESIDENT_LAYERS=0`（默认）、`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`（脚本默认导出）；改用 `H3_DIT_QUANT=w6a8` 重新下载；降低分辨率/时长（`SHORT_EDGE`、`DURATION`）。VAE 解码阶段 OOM 时确认没有把 `vae` 加入 `--layerwise-offload-components`（脚本默认不加）。

**Q：生成很慢？**
量化 offload 方案速度取决于主机内存与磁盘：内存 ≥64GB 时权重可 pin 驻，接近计算墙；32GB 主机主要靠 NVMe 流式传输。请使用 PCIe NVMe SSD 并关闭占内存的其他进程。参考量级：官方 4090 在 768p/107 帧/20 步下约 5 分钟一条（INT8+FA）。

**Q：音频可以作为唯一参考吗？**
不可以。音频必须与图像/视频搭配；混合输入总计 ≤12 个文件，图像 ≤9 张，视频 ≤3 段且总时长 ≤15s。

**Q：HuggingFace 模型需要申请（gated）怎么办？**
本模板全部走魔搭社区：`MiniMax/MiniMax-H3` 与 `Comfy-Org/MiniMax-H3` 均公开可直接下载，无需 token。

**Q：有 4 卡 H100/H200，想用全精度？**
`H3_WEIGHTS=online bash scripts/download-model.sh` 下载官方 BF16 全量，多卡时 `serve.sh` 自动切回全精度 `--performance-mode speed` 配方（默认 Ulysses N）；4×H100(80GB) 可用 `SGLANG_TP_SIZE=2 SGLANG_ULYSSES_DEGREE=2`。

**Q：输出时长 / 分辨率怎么调？**
`DURATION`（4–15 秒）、`SHORT_EDGE`（768）、`ASPECT_RATIO`（`auto` 或 21:9/16:9/4:3/1:1/3:4/9:16）、`SEED` 均可作为环境变量传给 `request-ref2va.sh`。低显存档位建议先用短时长、480p 左右验证链路。

**Q：许可证有什么限制？**
MiniMax-H3 权重采用社区许可证：商业使用免费但须在 UI 中展示 “MiniMax H3”，年收入超 2000 万美元需书面授权；适用区域不包含美国、欧盟、英国、韩国（详见模型 LICENSE）。中国大陆使用不受该地域条款限制。

## 参考链接

- MiniMax-H3 官方仓库（中文 README）：https://github.com/MiniMax-AI/MiniMax-H3/blob/main/README.zh-CN.md
- SGLang × MiniMax-H3 Cookbook（含消费级 GPU 实测章节）：https://docs.sglang.io/cookbook/diffusion/MiniMax/MiniMax-H3
- SGLang Diffusion 量化文档（ConvRot INT8 / NVFP4-AWQ）：https://docs.sglang.io/docs/sglang-diffusion/quantization
- 魔搭量化权重仓库：https://modelscope.cn/models/Comfy-Org/MiniMax-H3
- 魔搭官方权重：https://www.modelscope.cn/models/MiniMax/MiniMax-H3
- 开源权重硬件成本分析（42.5GB 组合的由来）：https://www.atlascloud.ai/zh/blog/tips/minimax-h3-open-source-weights
- CloudStudio 应用创建：https://ide.cloud.tencent.com/docs/guide/quick_start/developer/create-your-app/
- CloudStudio 运行配置 preview.yml：https://ide.cloud.tencent.com/docs/guide/quick_start/developer/how-to-run/
