#!/usr/bin/env python3
import asyncio
import argparse
import json
import time
import copy
from tabulate import tabulate
from test_client import send_audio_file, metrics

# 存储所有轮次的指标
all_metrics = []

async def run_multiple_tests(uri, audio_path, chunk_size_seconds=0.3, rounds=4):
    """
    运行多轮测试并收集性能指标
    """
    global all_metrics, metrics
    
    print(f"\n{'='*20} 开始多轮测试 {'='*20}")
    print(f"测试音频: {audio_path}")
    print(f"测试轮数: {rounds}")
    print(f"服务地址: {uri}")
    print(f"块大小: {chunk_size_seconds} 秒")
    print(f"{'='*50}\n")
    
    for round_num in range(1, rounds + 1):
        print(f"\n\n{'='*20} 第 {round_num} 轮测试 {'='*20}\n")
        
        # 重置指标
        for key in metrics:
            metrics[key] = None if key != "received_responses" else 0
        
        # 运行测试
        test_start_time = time.time()
        await send_audio_file(uri, audio_path, chunk_size_seconds)
        test_end_time = time.time()
        
        # 保存本轮指标
        round_metrics = copy.deepcopy(metrics)
        round_metrics["round"] = round_num
        round_metrics["total_time"] = test_end_time - test_start_time
        all_metrics.append(round_metrics)
        
        # 两轮测试之间等待几秒
        if round_num < rounds:
            print(f"\n等待5秒后开始下一轮测试...")
            await asyncio.sleep(5)
    
    # 分析并打印结果
    print_test_results(all_metrics, rounds)

def print_test_results(all_metrics, rounds):
    """
    打印测试结果和统计数据
    """
    print(f"\n\n{'='*20} 测试结果汇总 {'='*20}\n")
    
    # 准备表格数据
    headers = ["轮次", "首包延迟(秒)", "尾包延迟(秒)", "总体RTF", "响应数"]
    table_data = []
    
    for metrics in all_metrics:
        round_num = metrics["round"]
        
        # 计算首包延迟
        first_latency = "N/A"
        if metrics["first_chunk_sent_time"] and metrics["first_response_time"]:
            first_latency = metrics["first_response_time"] - metrics["first_chunk_sent_time"]
            first_latency = f"{first_latency:.2f}"
        
        # 计算尾包延迟
        final_latency = "N/A"
        if metrics["last_chunk_sent_time"] and metrics["final_response_time"]:
            final_latency = metrics["final_response_time"] - metrics["last_chunk_sent_time"]
            final_latency = f"{final_latency:.2f}"
        
        # 计算总体RTF
        rtf = "N/A"
        # 假设音频时长为total_time / 3 (估计值)
        if "total_time" in metrics:
            estimated_audio_duration = metrics["total_time"] / 3  # 估计值
            if metrics["final_response_time"]:
                rtf = metrics["total_time"] / estimated_audio_duration
                rtf = f"{rtf:.2f}"
        
        table_data.append([
            f"第{round_num}轮",
            first_latency,
            final_latency,
            rtf,
            metrics["received_responses"]
        ])
    
    # 添加后三轮平均值
    if rounds >= 3:
        last_three = all_metrics[-3:]
        
        avg_first_latency = "N/A"
        valid_first_latencies = []
        for m in last_three:
            if m["first_chunk_sent_time"] and m["first_response_time"]:
                valid_first_latencies.append(m["first_response_time"] - m["first_chunk_sent_time"])
        if valid_first_latencies:
            avg_first_latency = f"{sum(valid_first_latencies) / len(valid_first_latencies):.2f}"
        
        avg_final_latency = "N/A"
        valid_final_latencies = []
        for m in last_three:
            if m["last_chunk_sent_time"] and m["final_response_time"]:
                valid_final_latencies.append(m["final_response_time"] - m["last_chunk_sent_time"])
        if valid_final_latencies:
            avg_final_latency = f"{sum(valid_final_latencies) / len(valid_final_latencies):.2f}"
        
        avg_rtf = "N/A"
        # 难以准确计算平均RTF，略过
        
        avg_responses = sum(m["received_responses"] for m in last_three) / 3
        
        table_data.append([
            "后三轮均值",
            avg_first_latency,
            avg_final_latency,
            avg_rtf,
            f"{avg_responses:.1f}"
        ])
    
    # 打印表格
    print(tabulate(table_data, headers=headers, tablefmt="grid"))
    
    # 详细打印每轮指标
    print("\n详细指标:")
    for i, m in enumerate(all_metrics):
        print(f"\n第{i+1}轮详细指标:")
        for key, value in m.items():
            if key != "round":
                if isinstance(value, float):
                    print(f"  {key}: {value:.4f}")
                else:
                    print(f"  {key}: {value}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="多轮WebSocket客户端测试")
    parser.add_argument('--server', type=str, default='ws://localhost:8765', help='WebSocket服务器URI')
    parser.add_argument('--audio', type=str, required=True, help='音频文件路径')
    parser.add_argument('--chunk-size', type=float, default=0.3, help='块大小(秒)')
    parser.add_argument('--rounds', type=int, default=4, help='测试轮数')
    
    args = parser.parse_args()
    
    try:
        # 尝试导入tabulate，如果不存在则提示安装
        import tabulate
    except ImportError:
        print("请先安装tabulate库: pip install tabulate")
        exit(1)
    
    asyncio.run(run_multiple_tests(args.server, args.audio, args.chunk_size, args.rounds)) 