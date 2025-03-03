#!/usr/bin/env python3
import asyncio
import websockets
import json
import base64
import argparse
import numpy as np
import soundfile as sf
import time

async def send_audio_file(uri, audio_path, chunk_size_seconds=0.3):
    """
    将音频文件分块发送到WebSocket服务器
    
    参数:
        uri: WebSocket服务器URI
        audio_path: 音频文件路径
        chunk_size_seconds: 每个块的大小（秒）
    """
    # 加载音频文件
    audio, sample_rate = sf.read(audio_path)
    
    # 确保音频是单声道、float32格式
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)  # 转换为单声道
    
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32)
    
    # 重采样到16kHz（如果需要）
    if sample_rate != 16000:
        try:
            from librosa import resample
            audio = resample(audio, orig_sr=sample_rate, target_sr=16000)
            sample_rate = 16000
        except ImportError:
            print("警告: librosa 未安装，无法重采样到16kHz。这可能会影响识别质量。")
    
    # 计算每个块的样本数
    chunk_size = int(chunk_size_seconds * sample_rate)
    
    # 连接到WebSocket服务器，增加超时和重试
    max_retries = 3
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            async with websockets.connect(uri, ping_interval=20, ping_timeout=20) as websocket:
                # 开始会话
                session_id = f"test_{time.time()}"
                await websocket.send(json.dumps({
                    "command": "start_session",
                    "session_id": session_id
                }))
                
                response = await websocket.recv()
                print(f"Session started: {response}")
                
                # 分块发送音频
                for i in range(0, len(audio), chunk_size):
                    chunk = audio[i:i+chunk_size]
                    
                    # 将音频块转换为bytes并进行base64编码
                    chunk_bytes = chunk.tobytes()
                    chunk_base64 = base64.b64encode(chunk_bytes).decode('utf-8')
                    
                    # 发送音频块
                    await websocket.send(json.dumps({
                        "command": "process_audio",
                        "audio": chunk_base64
                    }))
                    
                    try:
                        # 接收转录结果，增加超时处理
                        response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                        print(f"Received: {response}")
                    except asyncio.TimeoutError:
                        print("Warning: Server response timeout, continuing...")
                    
                    # 模拟实时处理
                    await asyncio.sleep(chunk_size_seconds)
                
                # 结束会话
                await websocket.send(json.dumps({
                    "command": "end_session"
                }))
                
                try:
                    # 接收最终结果
                    response = await asyncio.wait_for(websocket.recv(), timeout=10.0)
                    print(f"Final result: {response}")
                    
                    # 接收会话结束确认
                    response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                    print(f"Session ended: {response}")
                except asyncio.TimeoutError:
                    print("Warning: Final response timeout")
                
                # 成功完成，跳出循环
                break
                
        except (websockets.exceptions.ConnectionClosedError, 
                websockets.exceptions.InvalidStatusCode, 
                ConnectionRefusedError) as e:
            retry_count += 1
            if retry_count >= max_retries:
                print(f"Error: Failed to connect after {max_retries} attempts: {e}")
                raise
            print(f"Connection error: {e}. Retrying ({retry_count}/{max_retries})...")
            await asyncio.sleep(2)  # 等待2秒后重试

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WebSocket Client for Whisper Streaming")
    parser.add_argument('--server', type=str, default='ws://localhost:8765', help='WebSocket server URI')
    parser.add_argument('--audio', type=str, required=True, help='Path to the audio file')
    parser.add_argument('--chunk-size', type=float, default=0.3, help='Chunk size in seconds')
    
    args = parser.parse_args()
    
    asyncio.run(send_audio_file(args.server, args.audio, args.chunk_size)) 