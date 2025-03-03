
# Whisper Streaming WebSocket 服务

## 简介

Whisper Streaming WebSocket 服务是一个基于 OpenAI Whisper 模型的实时语音转文字服务。它通过 WebSocket 协议提供流式语音识别功能，支持多种语言，并可以通过配置文件调整各种参数以优化识别效果。

## 目录

1. [安装与部署](#安装与部署)
2. [配置说明](#配置说明)
3. [API 接口](#api-接口)
4. [客户端示例](#客户端示例)
5. [常见问题](#常见问题)
6. [性能优化](#性能优化)
7. [故障排除](#故障排除)

## 安装与部署

### 系统要求

- Docker 和 Docker Compose
- NVIDIA GPU (推荐用于更快的处理速度)
- NVIDIA Container Toolkit (用于 GPU 支持)

### 使用 Docker 部署

1. **克隆代码仓库**

   ```bash
   git clone https://github.com/yourusername/whisper-streaming.git
   cd whisper-streaming
   ```

2. **构建 Docker 镜像**

   ```bash
   docker-compose build
   ```

3. **启动服务**

   ```bash
   docker-compose up -d
   ```

4. **查看日志**

   ```bash
   docker-compose logs -f
   ```

### 手动安装

1. **安装依赖**

   ```bash
   pip install -r requirements.txt
   ```

2. **启动服务**

   ```bash
   python server.py --config config.yaml
   ```

## 配置说明

服务配置通过 `config.yaml` 文件进行管理。以下是主要配置项：

### 基本配置

```
backend: "faster-whisper"  # 后端引擎，支持 "faster-whisper", "whisper_timestamped", "mlx-whisper", "openai-api"
task: "transcribe"         # 任务类型，支持 "transcribe" 或 "translate"
vad: false                 # 是否启用语音活动检测
log_level: "INFO"          # 日志级别，支持 "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"
```

### 模型配置

```
# 模型配置
model: "large-v3-turbo"    # 模型大小，支持 tiny, base, small, medium, large 等
lan: "auto"                # 语言代码，如 en, zh, fr 等，或 "auto" 自动检测

# 性能选项
device: "cuda"             # 计算设备，支持 "cpu" 或 "cuda"
compute_type: "float32"    # 计算精度，支持 "float32", "float16", "int8"

# 模型路径（可选）
model_cache_dir: "./models"  # 模型缓存目录
model_dir: "large-v3-turbo"  # 模型目录
```

### 处理配置

```
# 处理配置
buffer_trimming: "segment"   # 缓冲区修剪策略，支持 "segment" 或 "sentence"
buffer_trimming_sec: 0.3     # 缓冲区修剪长度阈值（秒）
vac: false                   # 是否启用语音活动控制器
min_chunk_size: 0.3          # 最小音频块大小（秒）

# 生成参数控制重复
repetition_penalty: 1.5      # 重复惩罚系数，值 > 1.0 会抑制重复
no_repeat_ngram_size: 3      # 防止重复的 n-gram 大小
```

## API 接口

服务通过 WebSocket 协议提供 API 接口，默认端口为 8765。

### 连接

```
ws://your-server-address:8765
```

### 消息格式

所有消息均使用 JSON 格式。

### 命令

#### 1. 开始会话

**请求**:
```
{
  "command": "start_session",
  "session_id": "optional_custom_session_id"
}
```

**响应**:
```
{
  "type": "session_started",
  "session_id": "session_id"
}
```

#### 2. 处理音频

**请求**:
```
{
  "command": "process_audio",
  "audio": "base64_encoded_audio_data"
}
```

**响应**:
```
{
  "type": "transcription",
  "start": 1.5,
  "end": 3.2,
  "text": "转录文本",
  "is_final": false
}
```

#### 3. 结束会话

**请求**:
```
{
  "command": "end_session"
}
```

**响应**:
```
{
  "type": "session_ended",
  "session_id": "session_id"
}
```

### 错误处理

当发生错误时，服务器会返回错误消息：

```
{
  "type": "error",
  "message": "错误信息"
}
```

## 客户端示例

### Python 客户端

```python
import asyncio
import websockets
import json
import base64
import numpy as np
import soundfile as sf

async def transcribe_audio(audio_path):
    uri = "ws://localhost:8765"
    
    # 加载音频
    audio, sample_rate = sf.read(audio_path)
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)
    
    # 分块大小（秒）
    chunk_size_seconds = 0.3
    chunk_size = int(chunk_size_seconds * sample_rate)
    
    async with websockets.connect(uri) as websocket:
        # 开始会话
        await websocket.send(json.dumps({
            "command": "start_session"
        }))
        response = await websocket.recv()
        print(f"Session started: {response}")
        
        # 分块发送音频
        for i in range(0, len(audio), chunk_size):
            chunk = audio[i:i+chunk_size]
            chunk_bytes = chunk.tobytes()
            chunk_base64 = base64.b64encode(chunk_bytes).decode('utf-8')
            
            await websocket.send(json.dumps({
                "command": "process_audio",
                "audio": chunk_base64
            }))
            
            response = await websocket.recv()
            print(f"Received: {response}")
            
            # 模拟实时处理
            await asyncio.sleep(chunk_size_seconds)
        
        # 结束会话
        await websocket.send(json.dumps({
            "command": "end_session"
        }))
        
        response = await websocket.recv()
        print(f"Final result: {response}")
        
        response = await websocket.recv()
        print(f"Session ended: {response}")

# 运行示例
asyncio.run(transcribe_audio("your_audio_file.wav"))
```

### JavaScript 客户端

```javascript
// 浏览器环境中使用 WebSocket 进行语音识别
async function transcribeAudio(audioBlob) {
    // 创建 WebSocket 连接
    const socket = new WebSocket('ws://your-server-address:8765');
    
    // 等待连接建立
    await new Promise(resolve => {
        socket.onopen = resolve;
    });
    
    // 开始会话
    socket.send(JSON.stringify({
        command: 'start_session'
    }));
    
    // 处理响应
    socket.onmessage = function(event) {
        const response = JSON.parse(event.data);
        console.log('Received:', response);
        
        if (response.type === 'transcription') {
            // 处理转录结果
            document.getElementById('transcript').textContent += response.text + ' ';
        }
    };
    
    // 将音频分块并发送
    const chunkSize = 4800; // 0.3秒 @ 16kHz
    const audioBuffer = await audioBlob.arrayBuffer();
    const audioData = new Float32Array(audioBuffer);
    
    for (let i = 0; i < audioData.length; i += chunkSize) {
        const chunk = audioData.slice(i, i + chunkSize);
        const base64Chunk = btoa(String.fromCharCode.apply(null, new Uint8Array(chunk.buffer)));
        
        socket.send(JSON.stringify({
            command: 'process_audio',
            audio: base64Chunk
        }));
        
        // 等待一小段时间，模拟实时处理
        await new Promise(resolve => setTimeout(resolve, 300));
    }
    
    // 结束会话
    socket.send(JSON.stringify({
        command: 'end_session'
    }));
    
    // 关闭连接
    setTimeout(() => {
        socket.close();
    }, 1000);
}

// 使用示例
document.getElementById('recordButton').addEventListener('click', async () => {
    // 获取麦克风权限并录制音频
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const mediaRecorder = new MediaRecorder(stream);
    const audioChunks = [];
    
    mediaRecorder.addEventListener('dataavailable', event => {
        audioChunks.push(event.data);
    });
    
    mediaRecorder.addEventListener('stop', async () => {
        const audioBlob = new Blob(audioChunks);
        await transcribeAudio(audioBlob);
    });
    
    // 开始录制 5 秒
    mediaRecorder.start();
    setTimeout(() => mediaRecorder.stop(), 5000);
});
```

## 常见问题

### 1. 如何提高识别准确率？

- 使用更大的模型（如 large-v3-turbo）
- 指定正确的语言代码，而不是使用自动检测
- 调整 `repetition_penalty` 和 `no_repeat_ngram_size` 参数
- 使用高质量的音频输入，减少背景噪音

### 2. 如何减少延迟？

- 减小 `min_chunk_size` 和 `buffer_trimming_sec` 值
- 使用较小的模型（如 small 或 medium）
- 使用 GPU 加速
- 使用 `float16` 或 `int8` 计算精度

### 3. 如何处理多语言识别？

- 设置 `lan: "auto"` 启用语言自动检测
- 对于已知语言，指定具体的语言代码以提高准确率
- 如需翻译，设置 `task: "translate"`

### 4. 如何解决重复文本问题？

- 增加 `repetition_penalty` 值（1.5-3.0 之间）
- 设置适当的 `no_repeat_ngram_size`（通常为 2-4）

## 性能优化

### 服务器优化

1. **GPU 加速**
   - 确保使用 CUDA 支持
   - 使用 `compute_type: "float16"` 减少内存使用

2. **内存管理**
   - 定期清理不活跃的会话
   - 限制并发会话数量

3. **模型缓存**
   - 使用 `model_cache_dir` 缓存模型文件
   - 预加载常用模型

### 客户端优化

1. **音频预处理**
   - 确保使用 16kHz 采样率
   - 使用单声道音频
   - 降噪和音量归一化

2. **分块策略**
   - 使用合适的分块大小（0.3-1.0 秒）
   - 考虑使用重叠分块以提高连续性

## 故障排除

### 连接问题

- 检查服务器是否正在运行
- 确认防火墙设置允许 WebSocket 连接
- 验证服务器地址和端口是否正确

### 识别问题

- 检查音频格式是否正确（16kHz, 单声道, float32）
- 确认音频数据的 base64 编码是否正确
- 查看服务器日志以获取详细错误信息

### 性能问题

- 检查 GPU 使用情况
- 监控内存使用
- 调整配置参数以平衡准确率和速度

### Docker 相关问题

- 确保 NVIDIA Container Toolkit 正确安装
- 检查 Docker 容器日志
- 验证挂载卷是否正确配置

