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
from datetime import datetime

# 性能指标记录
metrics = {
    "first_chunk_sent_time": None,
    "first_response_time": None,
    "last_chunk_sent_time": None,
    "final_response_time": None,
    "received_responses": 0,
}

# 事件标志
first_response_received = asyncio.Event()

async def send_audio_file(uri, audio_path, chunk_size_seconds=0.3):
    """
    将音频文件分块发送到WebSocket服务器
    
    参数:
        uri: WebSocket服务器URI
        audio_path: 音频文件路径
        chunk_size_seconds: 每个块的大小（秒）
    """
    global metrics
    
    # 输出版本信息
    import websockets
    print(f"Using websockets version: {websockets.__version__}")
    
    # 检查音频文件是否存在
    if not os.path.exists(audio_path):
        print(f"Error: Audio file not found: {audio_path}")
        return
    
    print(f"Loading audio file: {audio_path}")
    test_start_time = time.time()
    
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
        
        # 计算音频总时长（秒）
        audio_duration = len(audio) / sample_rate
        print(f"Audio duration: {audio_duration:.2f} seconds")
        
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
                
                # 重置事件标志
                first_response_received.clear()
                
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
                    for i in range(2):
                        response = await asyncio.wait_for(websocket.recv(), timeout=10.0)
                        response_data = json.loads(response)
                        print(f"Final response: {json.dumps(response_data, ensure_ascii=False)}")
                        
                        # 记录最终响应时间（第一个final response）
                        if i == 0 and metrics["final_response_time"] is None:
                            metrics["final_response_time"] = time.time()
                except asyncio.TimeoutError:
                    print("Warning: Final response timeout")
                
                # 输出性能指标
                print("\n===== 性能指标 =====")
                test_end_time = time.time()
                print(f"测试总时长: {test_end_time - test_start_time:.2f} 秒")
                print(f"音频总时长: {audio_duration:.2f} 秒")
                
                if metrics["first_chunk_sent_time"] and metrics["first_response_time"]:
                    first_latency = metrics["first_response_time"] - metrics["first_chunk_sent_time"]
                    print(f"首包延迟 (第一个音频块发送完成到收到第一个响应): {first_latency:.2f} 秒")
                
                if metrics["last_chunk_sent_time"] and metrics["final_response_time"]:
                    final_latency = metrics["final_response_time"] - metrics["last_chunk_sent_time"]
                    print(f"尾包延迟 (最后一个音频块发送完成到收到最终结果): {final_latency:.2f} 秒")
                
                if audio_duration > 0 and metrics["final_response_time"] and test_start_time:
                    overall_rtf = (metrics["final_response_time"] - test_start_time) / audio_duration
                    print(f"总体实时率 (RTF): {overall_rtf:.2f}x")
                
                print(f"收到的响应总数: {metrics['received_responses']}")
                print("===================")
                
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
    global metrics
    
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
        sent_time = time.time()
        await websocket.send(message)
        
        # 记录第一个块发送的时间
        if chunk_num == 1 and metrics["first_chunk_sent_time"] is None:
            metrics["first_chunk_sent_time"] = sent_time
            print(f"First chunk sent at: {datetime.fromtimestamp(sent_time).strftime('%H:%M:%S.%f')[:-3]}")
        
        # 记录最后一个块发送的时间
        if chunk_num == total_chunks:
            metrics["last_chunk_sent_time"] = sent_time
            print(f"Last chunk sent at: {datetime.fromtimestamp(sent_time).strftime('%H:%M:%S.%f')[:-3]}")
            
            # 等待首包响应，以计算首包时延
            if not first_response_received.is_set():
                try:
                    # 最多等待5秒
                    await asyncio.wait_for(first_response_received.wait(), timeout=5.0)
                except asyncio.TimeoutError:
                    print("Warning: No response received after sending all chunks")

async def receive_responses(websocket):
    """异步接收服务器响应"""
    global metrics
    
    try:
        while True:
            # 接收前记录时间
            before_recv_time = time.time()
            before_recv_datetime = datetime.fromtimestamp(before_recv_time).strftime('%H:%M:%S.%f')[:-3]
            print(f"[LOG] 等待接收响应... - 开始时间: {before_recv_datetime}")
            
            response = await websocket.recv()
            
            # 接收后记录时间
            received_time = time.time()
            received_datetime = datetime.fromtimestamp(received_time).strftime('%H:%M:%S.%f')[:-3]
            
            response_data = json.loads(response)
            
            # 增加响应计数
            metrics["received_responses"] += 1
            
            # 记录第一个响应的时间
            if metrics["first_response_time"] is None and "type" in response_data and response_data["type"] == "transcription":
                metrics["first_response_time"] = received_time
                print(f"[LOG] 首包响应时间: {received_datetime}")
                
                # 设置首次响应事件
                first_response_received.set()
            
            # 记录末尾响应的时间（针对已确认的文本）
            if "type" in response_data and response_data["type"] == "transcription" and response_data.get("status") == "completed":
                metrics["final_response_time"] = received_time
            
            # 输出响应信息
            response_type = response_data.get("type", "unknown")
            is_final = response_data.get("is_final", False)
            status = response_data.get("status", "")
            text = response_data.get("text", "")
            if len(text) > 30:
                text = text[:30] + "..."
            
            # 根据响应类型添加不同的标签
            if response_type == "transcription":
                if status == "completed":
                    status_label = "[已确认]"
                elif status == "partial":
                    status_label = "[临时]"
                else:
                    status_label = ""
                print(f"[LOG] 接收到转录 - 时间: {received_datetime}, 状态: {status_label}, 文本: {text}")
            elif response_type == "full_transcription":
                print(f"[LOG] 接收到完整转录 - 时间: {received_datetime}, 文本: {text}")
            else:
                print(f"[LOG] 接收到响应 - 时间: {received_datetime}, 类型: {response_type}, 是否最终: {is_final}, 文本: {text}")
            
            print(f"Received: {json.dumps(response_data, ensure_ascii=False)}")
    except asyncio.CancelledError:
        # 任务被取消时退出
        cancel_time = time.time()
        cancel_datetime = datetime.fromtimestamp(cancel_time).strftime('%H:%M:%S.%f')[:-3]
        print(f"[LOG] 接收任务取消 - 时间: {cancel_datetime}")
        return
    except Exception as e:
        error_time = time.time()
        error_datetime = datetime.fromtimestamp(error_time).strftime('%H:%M:%S.%f')[:-3]
        print(f"[LOG] 接收错误 - 时间: {error_datetime}, 错误: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WebSocket Client for Whisper Streaming")
    parser.add_argument('--server', type=str, default='ws://localhost:8765', help='WebSocket server URI')
    parser.add_argument('--audio', type=str, required=True, help='Path to the audio file')
    parser.add_argument('--chunk-size', type=float, default=0.3, help='Chunk size in seconds')
    
    args = parser.parse_args()
    
    asyncio.run(send_audio_file(args.server, args.audio, args.chunk_size)) 