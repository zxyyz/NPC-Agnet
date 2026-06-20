# 控制台

本项目是一个本地 AI 角色控制台：

- LLM：通过 `llama.cpp` 加载 Qwen GGUF。
- TTS：通过 `onnxruntime-gpu` 加载 Chatterbox multilingual ONNX q4。
- Web UI：默认运行在 `http://127.0.0.1:8090/`。

## 目录结构

```text
src/        主程序和 TTS CUDA 服务
config/     本地配置和角色设定
models/     本地模型，不提交 Git
runtime/    llama.cpp 等运行时，不提交 Git
scripts/    辅助启动脚本
```

## 启动

双击根目录的 `start.bat`。

程序会隐藏控制台窗口，启动 Web 控制台，并自动打开浏览器。

## 依赖

```powershell
pip install -r requirements.txt
```

还需要本机可用的 NVIDIA CUDA 环境，以及 `runtime/llama.cpp/llama-server.exe`。

完整部署说明见 [DEPLOYMENT.md](DEPLOYMENT.md)。

## 配置

- 首次运行会生成 `config/agent_settings.json` 和 `config/agnet_persona.txt`。
- 示例文件是 `config/agent_settings.example.json` 和 `config/agnet_persona.example.txt`。
- 真实配置包含本机路径和个人角色设定，已经在 `.gitignore` 中排除。

## GitHub 提交

模型和运行时文件默认被 `.gitignore` 排除。提交前建议确认：

```powershell
git status --ignored
```
