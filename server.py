#!/usr/bin/env python3
import asyncio
import websockets
import json
import logging
import argparse
import yaml
import numpy as np
import base64
import time
import os
from whisper_streaming import whisper_online

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('whisper_server')

# 全局变量
asr = None
online_processor = None
clients = {}  # 存储客户端连接和处理器

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

async def process_audio_chunk(websocket, audio_chunk, session_id):
    """处理音频块并返回转录结果"""
    global clients, asr
    
    # 如果是新会话，创建新的处理器
    if session_id not in clients:
        logger.info(f"Creating new processor for session {session_id}")
        
        # 使用已存在的ASR模型创建处理器
        from whisper_streaming.whisper_online import create_processor_from_model
        processor = create_processor_from_model(asr, args)
        
        clients[session_id] = {
            "processor": processor,
            "last_activity": time.time(),
            "complete_transcript": "",  # 用于存储累积的完整转录
            "last_end_time": 0  # 存储最后一个已处理片段的结束时间
        }
    
    # 更新最后活动时间
    clients[session_id]["last_activity"] = time.time()
    processor = clients[session_id]["processor"]
    
    # 处理音频
    processor.insert_audio_chunk(audio_chunk)
    
    try:
        result = processor.process_iter()
        
        # 处理新的返回格式
        completed = result["completed"]
        the_rest = result["the_rest"]
        
        # 创建响应
        response = {
            "type": "transcription",
            "is_final": False
        }
        
        # 如果有已确认的文本，添加到累积转录中
        if completed and completed[0] is not None:
            beg, end, text = completed
            
            # 更新会话的完整转录
            if beg >= clients[session_id]["last_end_time"]:
                # 只有当开始时间大于等于上一段的结束时间时才添加
                # 避免重复添加已转录的内容
                clients[session_id]["complete_transcript"] += " " + text
                clients[session_id]["complete_transcript"] = clients[session_id]["complete_transcript"].strip()
                clients[session_id]["last_end_time"] = end
            
            response.update({
                "start": beg,
                "end": end,
                "text": text,
                "status": "completed"
            })
            await websocket.send(json.dumps(response))
            logger.debug(f"Sent completed transcription: {text}")
        
        # 如果有未确认的文本，作为临时结果发送
        if the_rest and the_rest[0] is not None:
            beg, end, text = the_rest
            response = {
                "type": "transcription",
                "start": beg,
                "end": end,
                "text": text,
                "status": "partial",
                "is_final": False
            }
            await websocket.send(json.dumps(response))
            logger.debug(f"Sent partial transcription: {text}")
        
        # 发送真正的完整累积转录
        full_text = clients[session_id]["complete_transcript"]
        
        # 如果有部分确认的转录，添加到完整文本中以提供最新的输出
        if the_rest and the_rest[0] is not None:
            display_text = full_text + " " + the_rest[2]
            display_text = display_text.strip()
        else:
            display_text = full_text
        
        if display_text:
            response = {
                "type": "full_transcription",
                "text": display_text,
                "is_final": False
            }
            await websocket.send(json.dumps(response))
            
    except Exception as e:
        logger.error(f"Error processing audio: {e}")
        response = {
            "type": "error",
            "message": str(e)
        }
        await websocket.send(json.dumps(response))

async def finalize_transcription(websocket, session_id):
    """完成转录并返回最终结果"""
    global clients
    
    if session_id in clients:
        processor = clients[session_id]["processor"]
        result = processor.finish()
        
        if result[0] is not None:
            beg, end, text = result
            
            # 添加到累积的完整转录
            if beg >= clients[session_id]["last_end_time"]:
                clients[session_id]["complete_transcript"] += " " + text
                clients[session_id]["complete_transcript"] = clients[session_id]["complete_transcript"].strip()
            
            # 发送最终结果
            response = {
                "type": "transcription",
                "start": beg,
                "end": end,
                "text": text,
                "is_final": True,
                "status": "completed"
            }
            await websocket.send(json.dumps(response))
            
            # 发送完整的累积转录
            full_response = {
                "type": "full_transcription",
                "text": clients[session_id]["complete_transcript"],
                "is_final": True
            }
            await websocket.send(json.dumps(full_response))
            
            logger.info(f"Final transcription for session {session_id}: {clients[session_id]['complete_transcript']}")
        
        # 清理会话
        del clients[session_id]
        logger.info(f"Session {session_id} closed and cleaned up")

async def handle_connection(websocket, path=None):
    """处理WebSocket连接"""
    session_id = None
    
    try:
        async for message in websocket:
            try:
                data = json.loads(message)
                command = data.get("command")
                
                if command == "start_session":
                    # 开始新会话
                    session_id = data.get("session_id", str(time.time()))
                    logger.info(f"Starting new session: {session_id}")
                    await websocket.send(json.dumps({
                        "type": "session_started",
                        "session_id": session_id
                    }))
                
                elif command == "process_audio":
                    # 处理音频数据
                    if not session_id:
                        raise ValueError("Session not started")
                    
                    audio_base64 = data.get("audio")
                    if not audio_base64:
                        raise ValueError("No audio data provided")
                    
                    # 解码音频数据
                    audio_bytes = base64.b64decode(audio_base64)
                    audio_np = np.frombuffer(audio_bytes, dtype=np.float32)
                    
                    await process_audio_chunk(websocket, audio_np, session_id)
                
                elif command == "end_session":
                    # 结束会话
                    if not session_id:
                        raise ValueError("Session not started")
                    
                    await finalize_transcription(websocket, session_id)
                    await websocket.send(json.dumps({
                        "type": "session_ended",
                        "session_id": session_id
                    }))
                    session_id = None
                
                else:
                    raise ValueError(f"Unknown command: {command}")
            
            except json.JSONDecodeError:
                logger.error("Invalid JSON message")
                await websocket.send(json.dumps({
                    "type": "error",
                    "message": "Invalid JSON message"
                }))
            
            except Exception as e:
                logger.error(f"Error handling message: {e}")
                await websocket.send(json.dumps({
                    "type": "error",
                    "message": str(e)
                }))
    
    except websockets.exceptions.ConnectionClosed:
        logger.info("Connection closed")
    
    finally:
        # 清理会话
        if session_id and session_id in clients:
            await finalize_transcription(websocket, session_id)

async def cleanup_inactive_sessions():
    """定期清理不活跃的会话"""
    global clients
    
    while True:
        await asyncio.sleep(60)  # 每分钟检查一次
        
        current_time = time.time()
        sessions_to_remove = []
        
        for session_id, session_data in clients.items():
            # 如果会话超过10分钟不活跃，则清理
            if current_time - session_data["last_activity"] > 600:
                sessions_to_remove.append(session_id)
        
        for session_id in sessions_to_remove:
            logger.info(f"Cleaning up inactive session: {session_id}")
            del clients[session_id]

async def start_server():
    """启动WebSocket服务器"""
    global asr, args
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="Whisper Streaming WebSocket Server")
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the configuration YAML file')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to bind the server')
    parser.add_argument('--port', type=int, default=8765, help='Port to bind the server')
    
    # 添加Whisper相关参数
    whisper_online.add_shared_args(parser)
    
    args = parser.parse_args()
    
    # 设置日志级别
    if hasattr(args, 'log_level'):
        logger.setLevel(getattr(logging, args.log_level))
    
    # 输出websockets版本信息
    import websockets
    logger.info(f"Using websockets version: {websockets.__version__}")
    
    # 加载配置
    if os.path.exists(args.config):
        config = load_config(args.config)
        # 将配置合并到args
        for key, value in config.items():
            if not hasattr(args, key) or getattr(args, key) is None:
                setattr(args, key, value)
    
    # 初始化ASR模型
    logger.info("Initializing ASR model...")
    asr, _ = whisper_online.asr_factory(args)
    logger.info("ASR model initialized")
    
    # 启动WebSocket服务器
    server = await websockets.serve(
        handle_connection, 
        args.host, 
        args.port,
        ping_interval=30,
        ping_timeout=10
    )
    
    logger.info(f"Server started at ws://{args.host}:{args.port}")
    
    # 启动清理任务
    asyncio.create_task(cleanup_inactive_sessions())
    
    await server.wait_closed()

if __name__ == "__main__":
    try:
        asyncio.run(start_server())
    except KeyboardInterrupt:
        logger.info("Server stopped by user") 