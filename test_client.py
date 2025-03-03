#!/usr/bin/env python3
import asyncio
import websockets
import json
import base64
import argparse
import numpy as np
import soundfile as sf
import time
import os

async def send_audio_file(uri, audio_path, chunk_size_seconds=0.3):
    """
    将音频文件分块发送到WebSocket服务器
    
    参数:
        uri: WebSocket服务器URI
        audio_path: 音频文件路径
        chunk_size_seconds: 每个块的大小（秒）
    """
    # 输出版本信息
    import websockets
    print(f"Using websockets version: {websockets.__version__}")
    
    # 检查音频文件是否存在
    if not os.path.exists(audio_path):
        print(f"Error: Audio file not found: {audio_path}")
        return
    
    print(f"Loading audio file: {audio_path}")
    
    try:
        # 加载音频文件
        audio, sample_rate = sf.read(audio_path)
        print(f"Audio loaded: {len(audio)} samples, {sample_rate} Hz")
        
        # 确保音频是单声道、float32格式
        if len(audio.shape) > 1:
            print(f"Converting from {audio.shape} to mono")
            audio = audio.mean(axis=1)  # 转换为单声道
        
        if audio.dtype != np.float32:
            print(f"Converting from {audio.dtype} to float32")
            audio = audio.astype(np.float32)
        
        # 重采样到16kHz（如果需要）
        if sample_rate != 16000:
            try:
                from librosa import resample
                print(f"Resampling from {sample_rate} Hz to 16000 Hz")
                audio = resample(y=audio, orig_sr=sample_rate, target_sr=16000)
                sample_rate = 16000
            except ImportError:
                print("警告: librosa 未安装，无法重采样到16kHz。这可能会影响识别质量。")
        
        # 计算每个块的样本数
        chunk_size = int(chunk_size_seconds * sample_rate)
        print(f"Chunk size: {chunk_size} samples ({chunk_size_seconds} seconds)")
        
        try:
            print(f"Connecting to {uri}...")
            async with websockets.connect(uri, ping_interval=20, ping_timeout=20, close_timeout=10) as websocket:
                print("Connection established!")
                
                # 开始会话
                session_id = f"test_{time.time()}"
                print(f"Starting session: {session_id}")
                
                await websocket.send(json.dumps({
                    "command": "start_session",
                    "session_id": session_id
                }))
                
                try:
                    response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                    response_data = json.loads(response)
                    print(f"Session started: {json.dumps(response_data, ensure_ascii=False)}")
                except asyncio.TimeoutError:
                    print("Error: Session start timeout")
                    return
                
                # 创建发送和接收任务
                send_task = asyncio.create_task(send_audio_chunks(websocket, audio, chunk_size, chunk_size_seconds))
                receive_task = asyncio.create_task(receive_responses(websocket))
                
                # 等待发送任务完成
                await send_task
                
                # 结束会话
                print("Ending session...")
                await websocket.send(json.dumps({
                    "command": "end_session"
                }))
                
                # 等待接收任务结束
                receive_task.cancel()
                try:
                    await receive_task
                except asyncio.CancelledError:
                    pass
                
                # 接收最终响应
                try:
                    # 接收最终结果和会话结束消息
                    for _ in range(2):
                        response = await asyncio.wait_for(websocket.recv(), timeout=10.0)
                        response_data = json.loads(response)
                        print(f"Final response: {json.dumps(response_data, ensure_ascii=False)}")
                except asyncio.TimeoutError:
                    print("Warning: Final response timeout")
                
        except (websockets.exceptions.ConnectionClosedError,
                ConnectionRefusedError) as e:
            print(f"Connection error: {e}")
        
        except Exception as e:
            print(f"Unexpected error: {e}")
            import traceback
            traceback.print_exc()
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

async def send_audio_chunks(websocket, audio, chunk_size, chunk_size_seconds):
    """异步发送音频块"""
    total_chunks = (len(audio) + chunk_size - 1) // chunk_size
    
    start_time = time.time()
    for i in range(0, len(audio), chunk_size):
        chunk_num = i // chunk_size + 1
        
        # 计算应该发送这个块的时间
        expected_time = start_time + (i / len(audio)) * (len(audio) / 16000)
        now = time.time()
        
        # 如果运行太快，等待到适当的时间
        if now < expected_time:
            await asyncio.sleep(expected_time - now)
        
        print(f"Sending chunk {chunk_num}/{total_chunks}...")
        
        chunk = audio[i:i+chunk_size]
        
        # 将音频块转换为bytes并进行base64编码
        chunk_bytes = chunk.tobytes()
        chunk_base64 = base64.b64encode(chunk_bytes).decode('utf-8')
        
        # 发送音频块
        message = json.dumps({
            "command": "process_audio",
            "audio": chunk_base64
        })
        await websocket.send(message)

async def receive_responses(websocket):
    """异步接收服务器响应"""
    try:
        while True:
            response = await websocket.recv()
            response_data = json.loads(response)
            print(f"Received: {json.dumps(response_data, ensure_ascii=False)}")
    except asyncio.CancelledError:
        # 任务被取消时退出
        return
    except Exception as e:
        print(f"Error receiving response: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WebSocket Client for Whisper Streaming")
    parser.add_argument('--server', type=str, default='ws://localhost:8765', help='WebSocket server URI')
    parser.add_argument('--audio', type=str, required=True, help='Path to the audio file')
    parser.add_argument('--chunk-size', type=float, default=0.3, help='Chunk size in seconds')
    
    args = parser.parse_args()
    
    asyncio.run(send_audio_file(args.server, args.audio, args.chunk_size)) 