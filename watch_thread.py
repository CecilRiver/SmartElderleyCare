import shlex
import socket
import string
import subprocess
import threading

import queue

import task_threads

import get_request as g

import config as conf

import time

import Threading
import task_threads

import multiprocessing

class JLikeMap:
    def __init__(self):
        self.value = list()
        self.key = list()

    def put(self, key, value):
        i = self.contains_key(key)
        if i >= 0:
            self.value[i] = value
        else:
            self.key.append(key)
            self.value.append(value)

    def get(self, key):
        i = self.contains_key(key)
        if i >= 0:
            return self.value[i]
        else:
            return None

    def contains_key(self, key):
        for i in range(0, len(self.key)):
            if self.key[i] == key:
                return i
        return -1

    def remove(self, key):
        i = self.contains_key(key)
        if i >= 0:
            self.key.remove(self.key[i])
            self.value.remove(self.value[i])

    def clear(self):
        self.value.clear()
        self.key.clear()

    def size(self):
        return len(self.key)

    def keys(self):
        return self.key

    def values(self):
        return self.value


class WatchThread(threading.Thread):
    def __init__(self, server_ip_port: string, ratio: int):
        super().__init__()
        # IP地址和端口，格式为“<IP>:<PORT>”
        self.server = server_ip_port
        # 抽帧比例
        self.ratio = ratio
        # TCP连接对象
        # self.client = TCPClient(self.server)
        # 连接
        # self.client.connect()
        # 任务线程列表
        self.tasks = JLikeMap()
        self._running = False

    def run(self):
        print("[Watch] I'm started")
        self._running = True
        while self._running:
            # 从服务端接受指令
            self.solve_message()
            # 接受任务返回信息
            for t in self.tasks.values():
                while t[2].qsize() > 0:
                    #print("$TASK_MESSAGE#" + str(t[0]) + "#" + str(t[2].get(timeout=0)))
                    t[2].get(timeout=0)
            time.sleep(1)
        print("[Watch] Bye...")



    def solve_message(self):
        # 接受消息，可能有多条
        r = g.get_json_response(f'{conf.MANAGE_SERVER_BASE}{conf.API_TASK_START}?token={conf.SELF_TOKEN}')
        # 分割消息
        try:
            if r['Success']:
                print("[Watch] TASK START MESSAGE!")
                if self.tasks.key.__contains__(r['task_id']):
                    print("[Watch] Task id [" + r['task_id'] + "] is already running")
                else:
                    print(r['task_type'])
                    if r['task_type'] == 1:
                        print(111111)
                        q = queue.Queue()
                        w = Threading.FaceMainThread(r['task_id'], q, r['device_downstream_rtmp'], r['task_upstream_rtmp'], self.ratio, 30)
                        w.daemon = True
                        w.start()
                        self.tasks.put(r['task_id'], [r['task_id'], w, q])
                        id = r['task_id']
                        g.get_json_response(f'{conf.MANAGE_SERVER_BASE}{conf.API_TASK_CONFIRM_START}?token={conf.SELF_TOKEN}&task_id={id}')
                        print("[Watch] Task with id [" + r['task_id'] + "] have started")
                    elif r['task_type'] == 2:
                        print(22222)
                        # q = queue.Queue()
                        # w = task_threads.DetectMainThread(r['task_id'], q, r['device_downstream_rtmp'], r['task_upstream_rtmp'], 1, 30)
                        # w.daemon = True
                        # w.start()
                        # #self.tasks.put(r['task_id'], [r['task_id'], w, q])
                        # p = multiprocessing.Process(target=w.run())
                        # p.start()
                        id = r['task_id']
                        python_script = "C:\GitAlgorithm\yolo_slowfast\yolo_deepsort.py"

                        # 传递给另一个脚本的命令行参数
                        args = [r['device_downstream_rtmp'], r['task_upstream_rtmp']]
                        venv_path = 'C:/GitAlgorithm/yolo_slowfast/venv'
                        python_executable = f'{venv_path}/Scripts/python'
                        # 构造完整的命令行执行命令
                        command = [python_executable, python_script] + args
                        # 执行命令
                        cmd = shlex.join(command).replace('\'', '')
                        print(cmd)
                        subprocess.Popen(cmd, shell=False, stdin=subprocess.PIPE)

                        g.get_json_response(f'{conf.MANAGE_SERVER_BASE}{conf.API_TASK_CONFIRM_START}?token={conf.SELF_TOKEN}&task_id={id}')
                        print("[Watch] Task with id [" + r['task_id'] + "] have started")
                    elif r['task_type'] == 3:
                        print(33333)
                        # q = queue.Queue()
                        # w = task_threads.DetectMainThread(r['task_id'], q, r['device_downstream_rtmp'], r['task_upstream_rtmp'], 1, 30)
                        # w.daemon = True
                        # w.start()
                        # #self.tasks.put(r['task_id'], [r['task_id'], w, q])
                        # p = multiprocessing.Process(target=w.run())
                        # p.start()
                        id = r['task_id']
                        python_script = "C:\GitAlgorithm\yolo_slowfast\yolo_deepsort_slowfast.py"

                        # 传递给另一个脚本的命令行参数
                        args = [r['device_downstream_rtmp'], r['task_upstream_rtmp']]
                        venv_path = 'C:/GitAlgorithm/yolo_slowfast/venv'
                        python_executable = f'{venv_path}/Scripts/python'
                        # 构造完整的命令行执行命令
                        command = [python_executable, python_script] + args
                        # 执行命令
                        cmd = shlex.join(command).replace('\'', '')
                        print(cmd)
                        subprocess.Popen(cmd, shell=False, stdin=subprocess.PIPE)

                        g.get_json_response(f'{conf.MANAGE_SERVER_BASE}{conf.API_TASK_CONFIRM_START}?token={conf.SELF_TOKEN}&task_id={id}')
                        print("[Watch] Task with id [" + r['task_id'] + "] have started")
        except Exception as e:
            pass
        
        r = g.get_json_response(f'{conf.MANAGE_SERVER_BASE}{conf.API_TASK_END}?token={conf.SELF_TOKEN}')
            # 0     head
            # 1     name/id
        try:
            if r['Success']:
                # print("[Watch] TASK END MESSAGE!")
                t = self.tasks.get(r['task_id'])
                print(t[0])
                print(t[1])
                print(t[2])
                t[1].stop()
                self.tasks.remove(r['task_id'])
                print("[Watch] Task [" + r['task_id'] + "] has stop")
            else:
                print("[Watch] Task [" + r['task_id'] + "] is not running or not exist")
        except Exception as e:
            pass


if __name__ == '__main__':
    watch = WatchThread("192.168.43.1:2333", 15)
    watch.daemon = True
    watch.start()
    print("[Main] 原神，启动！")
    while(watch._running):
        pass
