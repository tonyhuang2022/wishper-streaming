import argparse
import yaml
from whisper_streaming import whisper_online
import soundfile as sf
import numpy as np
import librosa
import logging
import time

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def load_audio(file_path):
    audio, sr = librosa.load(file_path, sr=16000)
    return audio.astype(np.float32)

def load_audio_chunk(audio, beg, end):
    beg_s = int(beg * 16000)
    end_s = int(end * 16000)
    return audio[beg_s:end_s]

def output_transcript(o, now=None):
    if now is None:
        now = time.time() - start
    if o[0] is not None:
        print(f"{now*1000:.4f} {o[0]*1000:.0f} {o[1]*1000:.0f} {o[2]}")

def main():
    parser = argparse.ArgumentParser(description="Test Whisper Streaming")
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration YAML file')
    parser.add_argument('--audio', type=str, required=True, help='Path to the audio file for inference')
    args = parser.parse_args()
    
    # 设置日志级别
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # 获取whisper_online的logger
    whisper_logger = logging.getLogger('whisper_online')
    whisper_logger.setLevel(logging.INFO)

    # 创建一个文件处理器，将日志输出到文件
    file_handler = logging.FileHandler('whisper_online.log')
    file_handler.setLevel(logging.INFO)
    
    # 创建一个格式器
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    
    # 将文件处理器添加到logger
    whisper_logger.addHandler(file_handler)

    # Load configuration
    config = load_config(args.config)

    # Create ASR and Online processor
    asr, online_processor = whisper_online.asr_factory(argparse.Namespace(**config))

    # Load audio
    audio = load_audio(args.audio)
    duration = len(audio) / 16000

    # Set chunk size
    min_chunk = config.get('min_chunk_size', 0.2)  # 默认为0.2秒，如果配置文件中没有指定

    global start
    start = time.time()
    beg = 0
    end = 0

    full_transcript = []

    while True:
        now = time.time() - start
        if now < end + min_chunk:
            time.sleep(min_chunk + end - now)
        end = time.time() - start
        a = load_audio_chunk(audio, beg, end)
        beg = end
        online_processor.insert_audio_chunk(a)

        try:
            o = online_processor.process_iter()
        except AssertionError as e:
            whisper_logger.error(f"assertion error: {e}")
        else:
            if o[2]:  # If there's transcribed text
                output_transcript(o)
                full_transcript.append(o[2])
        
        now = time.time() - start
        whisper_logger.debug(f"## last processed {end:.2f} s, now is {now:.2f}, the latency is {now-end:.2f}")

        if end >= duration:
            break

    # Final flush
    o = online_processor.finish()
    output_transcript(o, now=duration)
    if o[2]:
        full_transcript.append(o[2])

    print("\nFull Transcript:")
    print("".join(full_transcript))

if __name__ == "__main__":
    main()