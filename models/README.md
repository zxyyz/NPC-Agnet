# Models

模型文件体积较大，不提交到 Git。

推荐目录：

- `models/llm/` 放置 Qwen GGUF 模型。
- `models/tts/BricksDisplay-chatterbox-multilingual-ONNX-q4/` 放置 Chatterbox multilingual ONNX q4 模型。

控制台会自动扫描 `models/llm/**/*.gguf`，也可以在界面里选择其他 GGUF。
