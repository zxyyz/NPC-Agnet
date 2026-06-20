# 部署环境说明

本文档用于在另一台 Windows 电脑上复现当前控制台环境。

## 已验证环境

- OS：Windows 11
- Python：3.12.10
- GPU：NVIDIA GeForce RTX 5060 Laptop GPU，显存约 8GB
- NVIDIA Driver：596.36
- LLM Runtime：llama.cpp Windows CUDA 构建
- TTS Runtime：onnxruntime-gpu 1.27.0

## 目录准备

项目根目录保持如下结构：

```text
控制台项目/
├─ src/
├─ config/
├─ models/
│  ├─ llm/
│  └─ tts/
├─ runtime/
│  └─ llama.cpp/
├─ scripts/
├─ start.bat
├─ requirements.txt
└─ DEPLOYMENT.md
```

## Python 环境

推荐使用 Python 3.12。

```powershell
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

当前 `requirements.txt`：

```text
requests
numpy
tokenizers
onnxruntime-gpu
```

如果 `onnxruntime-gpu` 启动时报 CUDA/cuDNN 相关 DLL 缺失，优先更新 NVIDIA 显卡驱动；仍然缺失时，再补装 onnxruntime-gpu 对应版本要求的 CUDA/cuDNN 运行库。

## llama.cpp

需要下载 Windows CUDA 版 llama.cpp，并把可执行文件放到：

```text
runtime/llama.cpp/llama-server.exe
```

同目录还需要保留 llama.cpp 发布包里的相关 DLL，例如：

```text
ggml-cuda.dll
llama.dll
llama-common.dll
cublas64_*.dll
cublasLt64_*.dll
```

程序默认通过 `llama-server.exe` 启动 OpenAI 兼容接口，端口为 `8080`。

## LLM 模型

把 Qwen GGUF 模型放到：

```text
models/llm/
```

控制台会自动扫描：

```text
models/llm/**/*.gguf
```

默认推荐路径：

```text
models/llm/Qwen3.5-4B-Q4_K_M-GGUF/qwen3-5-4B-Q4_K_M.gguf
```

也可以在控制台左侧模型选择里切换其它 GGUF。

## TTS 模型

当前使用 Chatterbox multilingual ONNX q4，模型目录必须是：

```text
models/tts/BricksDisplay-chatterbox-multilingual-ONNX-q4/
```

该目录至少需要包含：

```text
onnx/
tokenizer.json
Cangjie5_TC.json
default_voice.wav
female_voice.wav
config.json
```

TTS 服务默认使用 CUDA ONNXRuntime，端口为 `8081`。

## 本地配置

首次运行后会生成：

```text
config/agent_settings.json
config/agnet_persona.txt
```

如果需要手动初始化，可以复制示例文件：

```powershell
copy config\agent_settings.example.json config\agent_settings.json
copy config\agnet_persona.example.txt config\agnet_persona.txt
```

注意：`agent_settings.json` 里可以使用相对模型路径，也可以使用绝对路径。项目迁移到新机器时，推荐使用相对路径。

## 启动

双击：

```text
start.bat
```

或命令行启动：

```powershell
.\start.bat
```

启动后访问：

```text
http://127.0.0.1:8090/
```

`start.bat` 会隐藏控制台窗口，并自动打开 Web UI。

## 端口

- 控制台 UI：`8090`
- LLM llama.cpp：`8080`
- TTS ONNXRuntime：`8081`

如果端口被占用，先关闭旧进程，或在源码中修改对应端口常量。

## 一键部署检查清单

1. 安装 Python 3.12。
2. 安装或更新 NVIDIA 驱动。
3. 克隆项目。
4. 创建虚拟环境并执行 `pip install -r requirements.txt`。
5. 放入 `runtime/llama.cpp/llama-server.exe` 和配套 DLL。
6. 放入 `models/llm/` 下的 GGUF 模型。
7. 放入 `models/tts/BricksDisplay-chatterbox-multilingual-ONNX-q4/` 下的 TTS 模型。
8. 双击 `start.bat`。

## 快速排错

查看端口占用：

```powershell
Get-NetTCPConnection -LocalPort 8090,8080,8081 -ErrorAction SilentlyContinue
```

查看相关进程：

```powershell
Get-CimInstance Win32_Process |
  Where-Object {
    $_.CommandLine -like '*agent_control_panel.py*' -or
    $_.CommandLine -like '*chatterbox_cuda_server.py*' -or
    $_.CommandLine -like '*llama-server*'
  } |
  Select-Object ProcessId,Name,CommandLine
```

测试 TTS CUDA 是否可用：

```powershell
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
```

输出中应包含：

```text
CUDAExecutionProvider
```
