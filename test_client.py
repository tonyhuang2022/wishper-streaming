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
    "stable_responses": 0,
    "unstable_responses": 0,
}

# 事件标志
first_response_received = asyncio.Event()

# 当前转录状态
transcript_state = {
    "stable_text": "",
    "unstable_text": "",
    "combined_text": ""
}

async def send_audio_file(uri, audio_path, chunk_size_seconds=0.3):
    """
    将音频文件分块发送到WebSocket服务器
    
    参数:
        uri: WebSocket服务器URI
        audio_path: 音频文件路径
        chunk_size_seconds: 每个块的大小（秒）
    """
    global metrics, transcript_state
    
    # 重置状态
    transcript_state = {
        "stable_text": "",
        "unstable_text": "",
        "combined_text": ""
    }
    
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
            print(f"Resampling from {sample_rate}Hz to 16000Hz")
            # 这里应该添加重采样代码，但为简单起见，我们跳过
            # 依赖librosa或其他库进行重采样
        
        # 计算块大小（样本数）
        chunk_size = int(chunk_size_seconds * 16000)
        print(f"Chunk size: {chunk_size} samples ({chunk_size_seconds} seconds)")
        
        # 连接到WebSocket服务器
        async with websockets.connect(uri) as websocket:
            print(f"Connected to server: {uri}")
            
            # 开始会话
            session_id = f"test_{time.time()}"
            await websocket.send(json.dumps({
                "command": "start_session",
                "session_id": session_id
            }))
            
            response = await websocket.recv()
            print(f"Session started: {response}")
            
            # 创建接收响应的任务
            receive_task = asyncio.create_task(receive_responses(websocket))
            
            # 发送音频块
            await send_audio_chunks(websocket, audio, chunk_size, chunk_size_seconds)
            
            # 结束会话
            await websocket.send(json.dumps({
                "command": "end_session"
            }))
            
            # 等待接收最终响应
            try:
                await asyncio.wait_for(receive_task, timeout=5.0)
            except asyncio.TimeoutError:
                print("Timeout waiting for final response.")
            
            # 取消接收任务
            receive_task.cancel()
            
            # 打印性能指标
            print("\n--- Performance Metrics ---")
            
            if metrics["first_chunk_sent_time"] and metrics["first_response_time"]:
                first_latency = metrics["first_response_time"] - metrics["first_chunk_sent_time"]
                print(f"First response latency: {first_latency:.3f}s")
            else:
                print("First response latency: N/A")
            
            if metrics["last_chunk_sent_time"] and metrics["final_response_time"]:
                final_latency = metrics["final_response_time"] - metrics["last_chunk_sent_time"]
                print(f"Final response latency: {final_latency:.3f}s")
            else:
                print("Final response latency: N/A")
            
            total_time = time.time() - test_start_time
            print(f"Total responses: {metrics['received_responses']} (Stable: {metrics['stable_responses']}, Unstable: {metrics['unstable_responses']})")
            print(f"Total test time: {total_time:.3f}s")
            
            # 打印最终转录结果
            print("\n--- Final Transcription ---")
            print(transcript_state["combined_text"])
            
    except Exception as e:
        print(f"Error: {e}")

async def send_audio_chunks(websocket, audio, chunk_size, chunk_size_seconds):
    """异步发送音频块"""
    global metrics
    
    total_chunks = (len(audio) + chunk_size - 1) // chunk_size
    
    start_time = time.time()
    start_datetime = datetime.fromtimestamp(start_time).strftime('%H:%M:%S.%f')[:-3]
    print(f"[LOG] 开始发送音频块 - 开始时间: {start_datetime}")
    
    for i in range(0, len(audio), chunk_size):
        chunk_num = i // chunk_size + 1
        current_time = time.time()
        current_datetime = datetime.fromtimestamp(current_time).strftime('%H:%M:%S.%f')[:-3]
        
        # 计算应该发送这个块的时间
        expected_time = start_time + (i / len(audio)) * (len(audio) / 16000)
        expected_datetime = datetime.fromtimestamp(expected_time).strftime('%H:%M:%S.%f')[:-3]
        
        # 计算等待时间
        wait_time = 0
        if current_time < expected_time:
            wait_time = expected_time - current_time
            print(f"[LOG] 块 {chunk_num}/{total_chunks} - 当前时间: {current_datetime}, 预期时间: {expected_datetime}, 等待: {wait_time:.3f}秒")
            await asyncio.sleep(wait_time)
        else:
            print(f"[LOG] 块 {chunk_num}/{total_chunks} - 当前时间: {current_datetime}, 预期时间: {expected_datetime}, 无需等待")
        
        chunk = audio[i:i+chunk_size]
        
        # 将音频块转换为bytes并进行base64编码
        chunk_bytes = chunk.tobytes()
        chunk_base64 = base64.b64encode(chunk_bytes).decode('utf-8')
        
        # 发送前记录时间
        before_send_time = time.time()
        before_send_datetime = datetime.fromtimestamp(before_send_time).strftime('%H:%M:%S.%f')[:-3]
        
        # 发送音频块
        message = json.dumps({
            "command": "process_audio",
            "audio": chunk_base64
        })
        await websocket.send(message)
        
        # 发送后记录时间
        sent_time = time.time()
        sent_datetime = datetime.fromtimestamp(sent_time).strftime('%H:%M:%S.%f')[:-3]
        duration_ms = (sent_time - before_send_time) * 1000
        
        print(f"[LOG] 块 {chunk_num}/{total_chunks} - 发送前: {before_send_datetime}, 发送后: {sent_datetime}, 耗时: {duration_ms:.2f}ms, 大小: {len(chunk)} 样本")
        
        # 记录第一个块发送的时间
        if chunk_num == 1 and metrics["first_chunk_sent_time"] is None:
            metrics["first_chunk_sent_time"] = sent_time
            print(f"[LOG] 首块发送时间: {sent_datetime}")
        
        # 记录最后一个块发送的时间
        if chunk_num == total_chunks:
            metrics["last_chunk_sent_time"] = sent_time
            print(f"[LOG] 尾块发送时间: {sent_datetime}")
            
            # 等待首包响应，以计算首包时延
            if not first_response_received.is_set():
                try:
                    # 最多等待5秒
                    print(f"[LOG] 等待首包响应...")
                    await asyncio.wait_for(first_response_received.wait(), timeout=5.0)
                    print(f"[LOG] 收到首包响应")
                except asyncio.TimeoutError:
                    print(f"[LOG] 警告: 发送所有块后仍未收到响应")
    
    end_time = time.time()
    end_datetime = datetime.fromtimestamp(end_time).strftime('%H:%M:%S.%f')[:-3]
    duration = end_time - start_time
    print(f"[LOG] 所有音频块发送完毕 - 结束时间: {end_datetime}, 总用时: {duration:.3f}秒")

async def receive_responses(websocket):
    """异步接收服务器响应"""
    global metrics, transcript_state
    
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
            
            # 处理转录响应
            if "type" in response_data and response_data["type"] == "transcription":
                # 记录第一个响应的时间
                if metrics["first_response_time"] is None:
                    metrics["first_response_time"] = received_time
                    print(f"[LOG] 首包响应时间: {received_datetime}")
                    
                    # 设置首次响应事件
                    first_response_received.set()
                
                # 记录最终响应的时间
                if response_data.get("is_final", False):
                    metrics["final_response_time"] = received_time
                
                # 处理稳定性信息
                stability = response_data.get("stability", "unknown")
                if stability == "stable":
                    metrics["stable_responses"] += 1
                    transcript_state["stable_text"] = response_data.get("full_text", "")
                elif stability == "unstable":
                    metrics["unstable_responses"] += 1
                    transcript_state["unstable_text"] = response_data.get("text", "")
                
                # 更新组合文本
                if response_data.get("full_text"):
                    transcript_state["combined_text"] = response_data.get("full_text")
                
                # 输出响应信息
                response_type = response_data.get("type", "unknown")
                is_final = response_data.get("is_final", False)
                text = response_data.get("text", "")
                if len(text) > 30:
                    text_preview = text[:30] + "..."
                else:
                    text_preview = text
                
                stability_str = f", 稳定性: {stability}"
                full_text = response_data.get("full_text", "")
                if len(full_text) > 50:
                    full_text_preview = full_text[:50] + "..."
                else:
                    full_text_preview = full_text
                
                print(f"[LOG] 接收到响应 - 时间: {received_datetime}, 类型: {response_type}, 是否最终: {is_final}{stability_str}")
                print(f"[LOG] 文本片段: {text_preview}")
                print(f"[LOG] 完整文本: {full_text_preview}")
            
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