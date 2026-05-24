# IndexTTS2.0 本地语音合成系统

基于 IndexTeam 的 IndexTTS2 改进版，支持**零样本语音克隆**与**情感控制**的本地 TTS 系统。

> ⚠ 本仓库仅包含源代码、配置和小文件。  
> **大型模型文件（gpt.pth, s2mel.pth）请通过下方说明自行下载配置。**

---

## 系统要求

| 项目 | 最低 | 推荐 |
|------|------|------|
| 系统 | Windows 10/11 64 位 | Windows 11 |
| Python | 3.10 | 3.10（已测试） |
| 内存 | 8 GB | 16 GB+ |
| 硬盘 | 10 GB 可用空间 | 20 GB+（含模型） |
| 显卡 | CPU 模式可用 | NVIDIA GPU 6GB+（CUDA 12+） |
| CUDA | — | CUDA 12.8 / Driver 13.2+ |

---

## 目录结构

```
IndexTTS2_Debug/
├── app.py                          # WebUI 入口 (Gradio + FastAPI)
├── run_infer_test.py               # CPU 推理测试脚本
├── start_index_tts2.bat            # 一键启动脚本
├── cy_app.cp310-win_amd64.pyd      # WebUI 编译模块
├── README.md                       # 本文件
│
├── indextts/                       # 核心推理引擎
│   ├── infer_v2.cp310-win_amd64.pyd  # v2 推理编译模块 (GPT→s2mel→BigVGAN)
│   ├── infer.py                      # v1.5 Python 源码（与 v2 不兼容）
│   ├── cli.py                        # 命令行入口
│   ├── gpt/                          # GPT 模型
│   ├── s2mel/                        # 语义→梅频谱解码器
│   ├── BigVGAN/                      # 声码器（备用实现）
│   ├── vqvae/                        # VQ-VAE 编解码器
│   └── utils/                        # 工具（分词、正则化、WebUI）
│
├── checkpoints/                    # 模型文件目录
│   ├── config.yaml                  # 模型配置
│   ├── bpe.model                    # BPE 分词器模型
│   ├── feat1.pt                     # 说话人嵌入矩阵
│   ├── feat2.pt                     # 情感嵌入矩阵
│   └── wav2vec2bert_stats.pt        # wav2vec2bert 统计量
│
└── examples/                       # 示例参考音频
    └── voice_01.wav
```

---

## 快速开始

### 1. 安装 Python 3.10

下载 [Python 3.10 (Windows amd64)](https://www.python.org/downloads/release/python-31011/)，安装时勾选 **"Add Python to PATH"**。

### 2. 克隆本仓库

```bash
git clone https://github.com/yizheng-max/IndexTTS2-Deploy.git
cd IndexTTS2-Deploy
```

### 3. 下载大型模型文件

将以下文件放入 `checkpoints/` 目录：

| 文件 | 大小 | 说明 |
|------|------|------|
| `gpt.pth` | ~3.2 GB | GPT 主模型 |
| `s2mel.pth` | ~1.1 GB | 语义→梅频谱解码器 |

> **获取方式**：请从 IndexTTS2 官方发布页或你的原始安装包中复制这两个文件。

### 4. 安装依赖

```bash
# 创建虚拟环境（推荐）
python -m venv venv
venv\Scripts\activate

# 安装 PyTorch (CUDA 12.x)
pip install torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124

# CPU 版（无显卡用这个）
pip install torch==2.5.1 torchaudio==2.5.1

# 安装其他依赖
pip install -r requirements.txt
```

**依赖清单（requirements.txt）：**

```txt
gradio>=5.0.0
fastapi>=0.110.0
uvicorn[standard]
transformers>=4.45.0
soundfile>=0.12.1
omegaconf>=2.3.0
sentencepiece>=0.2.0
jieba>=0.42.1
pypinyin>=0.50.0
librosa>=0.10.0
einops>=0.8.0
WeTextProcessing>=1.0.0
pynini>=2.1.0
```

> ⚠ `pynini` 需要 Visual C++ 运行库，详见 [WeTextProcessing 文档](https://github.com/wenet-e2e/WeTextProcessing)。

### 5. 运行

**方式一：WebUI（推荐）**

```bash
python app.py
```

浏览器打开 `http://127.0.0.1:7860`

**方式二：命令行**

```powershell
# CPU 模式
python run_infer_test.py

# CUDA 模式
python -c "
import os, sys
os.environ['HF_HOME'] = './checkpoints'
from indextts.infer_v2 import IndexTTS2
tts = IndexTTS2(cfg_path='checkpoints/config.yaml', model_dir='checkpoints', is_fp16=True, device='cuda')
tts.infer('examples/voice_01.wav', '你好，这是测试文本。', 'outputs/result.wav', verbose=True)
"
```

### 6. 首次启动说明

首次运行时，会自动从 HuggingFace Hub 下载以下模型（约 5-10 分钟，视网络而定）：

| 模型 | 用途 |
|------|------|
| `nvidia/bigvgan_v2_22khz_80band_256x` | 声码器（波形生成） |
| `amphion/MaskGCT` | 语义编解码器 |
| `funasr/campplus` | 说话人编码器 |

下载后的模型会缓存在 `checkpoints/hub/` 目录。

---

## WebUI 使用

1. 启动 WebUI 后访问 `http://127.0.0.1:7860`
2. **参考音频**：上传一段 3-10 秒的清晰人声（或使用 `examples/voice_01.wav`）
3. **合成文本**：输入要合成的文字（中英文混合支持）
4. **情感控制**（可选）：
   - 上传情感参考音频（`emo_audio_prompt`）
   - 或手动输入情感向量（`emo_vector`）
   - 或使用文本情感分析（`use_emo_text`）
5. 点击合成，等待生成

---

## 性能参考

| 配置 | 生成 7.7 秒音频 | 实时倍率 (RTF) |
|------|----------------|:---------:|
| CPU (i7-13700H) | ~47 秒 | 8.1x |
| RTX 4060 Laptop (FP16) | ~11 秒 | 1.4x |
| RTX 4090 (估算) | ~3-5 秒 | 0.4-0.7x |

---

## 模型自动下载脚本 (PowerShell)

如果你希望一键下载所有缺失的模型文件，运行以下脚本：

```powershell
# download_models.ps1
$checkpoints = "checkpoints"
$hub = "$checkpoints\hub"

# 设置 HF 镜像（可选，国内加速）
$env:HF_ENDPOINT = "https://hf-mirror.com"

# 下载 gpt.pth 和 s2mel.pth
# ⚠ 这两个文件需要从原始发布页获取
Write-Host "请将 gpt.pth 和 s2mel.pth 放入 checkpoints/ 目录"

# HF Hub 模型会自动下载（首次运行触发）
Write-Host "运行 Python 脚本触发模型下载..."
python -c "
from huggingface_hub import snapshot_download
snapshot_download('nvidia/bigvgan_v2_22khz_80band_256x', cache_dir='./checkpoints/hub')
snapshot_download('amphion/MaskGCT', cache_dir='./checkpoints/hub')
snapshot_download('funasr/campplus', cache_dir='./checkpoints/hub')
print('模型下载完成！')
"
```

---

## 常见问题

**Q: 启动报错 "Failed to load custom CUDA kernel for BigVGAN"**  
A: 不影响使用，会自动回退到 PyTorch 实现。

**Q: 端口 7860 被占用**  
A: 修改 `app.py` 中的端口设置，或关闭占用程序：
```powershell
netstat -ano | findstr :7860
taskkill /PID <进程ID> /F
```

**Q: 合成结果音质不理想**  
A: 参考音频建议：3-10 秒、清晰人声、无背景音乐、与目标说话人音色相近。

**Q: 显存不足 (OOM)**  
A: 确保使用 FP16 半精度（`is_fp16=True`），或在 CPU 上运行。

---

## 技术架构

```
文本 → TextNormalizer → BPE Tokenizer → GPT (Latent) → s2mel Decoder → Mel-Spectrogram → BigVGAN v2 → 音频

参考音频 → CAM++ Speaker Encoder ──→ GPT Conditioning (Conformer Perceiver)
参考音频 → MaskGCT Semantic Codec ──→ s2mel 条件输入
```

- **GPT**: 基于 GPT 的语音语言模型，24 层，1280 维
- **s2mel**: 语义到梅频谱的扩散解码器 (DiT)，13 层
- **BigVGAN v2**: NVIDIA 的 22kHz 声码器（80 频段）
- **MaskGCT**: 语义编解码器（语音离散化）
- **CAM++**: 说话人嵌入提取

---

## 致谢

- [IndexTeam / IndexTTS2](https://github.com/IndexTeam/IndexTTS2) — 原始项目
- [NVIDIA BigVGAN v2](https://huggingface.co/nvidia/bigvgan_v2_22khz_80band_256x) — 高性能声码器
- [Amphion MaskGCT](https://huggingface.co/amphion/MaskGCT) — 语义编解码器
- [FunASR CAM++](https://github.com/modelscope/FunASR) — 说话人编码
