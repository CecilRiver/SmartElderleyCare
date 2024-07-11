from kafka import KafkaConsumer
import time
import json
import subprocess
import os
import sys

def generate_rtsp_url(rtmp_input_url):
    # 提取最后一部分 URL 作为输出路径的一部分
    path_parts = rtmp_input_url.split('/')
    if len(path_parts) > 2 and path_parts[-1] == 'stream.m3u8':
        path = '/'.join(path_parts[-2:-1])  # 获取倒数第三和倒数第二部分
    else:
        path = '/'.join(path_parts[-1:])  # 获取倒数第二和最后一部分
    return f"rtsp://47.93.76.253:8554/{path}"

def start_consumer():
    while True:
        try:
            consumer = KafkaConsumer(
                'video_task_start',
                'video_task_end',
                bootstrap_servers='47.102.213.168:9092',
                auto_offset_reset='earliest',  # 可选：从最早的消息开始消费
                group_id='my-consumer-group',  # 消费者组ID
            )

            for msg in consumer:
                message = json.loads(msg.value.decode())
                task = message.get("task", {})
                camera = message.get("camera", {})

                task_id = task.get("id")
                task_type = task.get("task_type")
                rtmp_input_url = camera.get("url")
                rtmp_input_url = generate_rtsp_url(rtmp_input_url)
                rtmp_output_url = f"rtsp://47.93.76.253:8554/{task.get('url_string')}"

                print(f"Task ID: {task_id}, Task Type: {task_type}")
                print(f"RTMP Input URL: {rtmp_input_url}")
                print(f"RTMP Output URL: {rtmp_output_url}")

                # 检查 task_type 是否是数字并转换为整数
                if isinstance(task_type, str) and task_type.isdigit():
                    task_type = int(task_type)
                # 根据 task_type 决定要运行的脚本文件
                if task_type == 1:
                    script_name = "face_recognition_emotion.py"
                elif task_type == 2:
                    script_name = "invade_detection.py"
                elif task_type == 3:
                    script_name = "face_recognition_detection.py"
                elif task_type == 4:
                    script_name = "interaction.py"
                elif task_type == 5:
                    script_name = "fall_detection.py"
                else:
                    print(f"Error: Invalid task type {task_type}!")
                    continue

                # 获取当前虚拟环境的Python解释器路径
                venv_python = os.path.join(os.getcwd(), 'venv', 'Scripts', 'python.exe') if sys.platform == "win32" else os.path.join(os.getcwd(), 'venv', 'bin', 'python')

                # 构建命令行参数
                cmd = [venv_python, script_name, rtmp_input_url, rtmp_output_url, str(task_id)]

                # 启动新的 Python 脚本
                subprocess.Popen(cmd)

        except Exception as e:
            print(f"Error: {e}")
            print("Reconnecting in 5 seconds...")
            time.sleep(5)

if __name__ == "__main__":
    start_consumer()
