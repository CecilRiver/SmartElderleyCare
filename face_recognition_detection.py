import subprocess
import threading
from queue import Queue
from sql import insert_event_info
from qiniuConfig import upload_to_qiniu
import cv2
import numpy as np
import torch
import random
import dlib
from datetime import datetime
import argparse

class MyVideoCapture:
    def __init__(self, source):
        self.cap = cv2.VideoCapture(source)
        self.frame_queue = Queue(maxsize=1)
        self.read_thread = threading.Thread(target=self._read_frames, daemon=True)
        self._reading = True
        self.read_thread.start()

    def _read_frames(self):
        while self._reading and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            if not self.frame_queue.full():
                self.frame_queue.put(frame)
        self._reading = False

    def read(self):
        if not self.frame_queue.empty():
            return True, self.frame_queue.get()
        return False, None

    def get(self, prop_id):
        return self.cap.get(prop_id)

    def release(self):
        self._reading = False
        self.read_thread.join()
        self.cap.release()

class FaceRecognition:
    def __init__(self, predictor_path, model_path, face_feature_path, face_list, yolo_model_path):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(predictor_path)
        self.facerec = dlib.face_recognition_model_v1(model_path)
        self.face_feature_array = np.loadtxt(face_feature_path, delimiter=',')
        self.face_list = face_list
        self.descriptors = []
        self.faces = []
        self.yolo_model = torch.hub.load('ultralytics/yolov5', 'custom',
                                         path=yolo_model_path).to("cuda")
        self.yolo_model.conf = 0.7
        self.yolo_model.iou = 0.7
        self.coco_color_map = [[random.randint(0, 255) for _ in range(3)] for _ in range(80)]

    def compute_dst(self, feature_1, feature_2):
        feature_1 = np.array(feature_1)
        feature_2 = np.array(feature_2)
        dist = np.linalg.norm(feature_1 - feature_2)
        return dist

    def process_frame(self, frame, task_id):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        dets = self.detector(gray, 1)
        unknown_detected = False

        for value in dets:
            shape = self.predictor(gray, value)
            cv2.rectangle(frame, (value.left(), value.top()), (value.right(), value.bottom()), (0, 255, 0), 2)

            # YOLO 检测
            yolo_preds = self.yolo_model([frame], size=640)

            for i, (*box, cls, _) in enumerate(yolo_preds.pred[0]):
                if int(cls) != 0:  # 不是人物
                    continue

                x1, y1, x2, y2 = box
                text = '{} {}'.format(yolo_preds.names[int(cls)], i)
                color = self.coco_color_map[int(cls)]
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.putText(frame, text, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # 人脸识别
            face_descriptor = self.facerec.compute_face_descriptor(frame, shape)
            v = np.array(face_descriptor)

            flag = False
            for j in range(len(self.face_list)):
                if self.compute_dst(v, self.face_feature_array[j]) < 0.4:
                    cv2.putText(frame, self.face_list[j], (value.left(), value.top()), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 1, cv2.LINE_AA)
                    flag = True
                    break

            if not flag:
                cv2.putText(frame, "Unknown", (value.left(), value.top()), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 1, cv2.LINE_AA)
                unknown_detected = True

            for pt in shape.parts():
                pos = (pt.x, pt.y)
                cv2.circle(frame, pos, 1, color=(0, 255, 0))

            if flag:
                self.descriptors.append(v)
                self.faces.append(frame.copy())
            else:
                sign = True
                for i in range(len(self.descriptors)):
                    distance = self.compute_dst(self.descriptors[i], v)
                    if distance < 0.4:
                        if cv2.Laplacian(gray, cv2.CV_64F).var() > cv2.Laplacian(cv2.cvtColor(self.faces[i], cv2.COLOR_BGR2GRAY), cv2.CV_64F).var():
                            self.faces[i] = frame.copy()
                        sign = False
                        break
                if sign:
                    self.descriptors.append(v)
                    self.faces.append(frame.copy())

        # 保存图像和信息
        if unknown_detected:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"unknown_{timestamp}.jpg"
            cv2.imwrite(filename, frame)  # Save the frame
            print(f"Detected 'Unknown' face. Frame saved as '{filename}'.")
            image_url = upload_to_qiniu(filename)
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            event_desc = "Unknown person detected"
            insert_event_info(3, current_time, 'YF209', event_desc, 0, image_url, task_id)

        return frame

def detect_and_track_unknown(rtmp_input_url, rtmp_output_url, task_id):
    # 初始化视频捕捉器和人脸识别器
    video_capture = MyVideoCapture(rtmp_input_url)

    predictor_path = "Resources/shape_predictor_68_face_landmarks.dat"
    model_path = "Resources/dlib_face_recognition_resnet_model_v1.dat"
    face_feature_path = "Resources/featureMean/feature_all.csv"
    face_list = ["shijiaxiang", "zhangkaige"]
    yolo_model_path = "C:/GitAlgorithm/yolo_slowfast/yolov5l6.pt"  # 模型路径根据实际情况修改
    face_recognition = FaceRecognition(predictor_path, model_path, face_feature_path, face_list, yolo_model_path)

    # FFmpeg推流
    frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    ffmpeg_path = "ffmpeg"  # FFmpeg可执行文件路径
    command = [
        ffmpeg_path,
        '-y', '-an',
        '-f', 'rawvideo',
        '-vcodec', 'rawvideo',
        '-pix_fmt', 'bgr24',
        '-s', "{}x{}".format(frame_width, frame_height),
        '-r', str(fps),
        '-i', '-',
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-preset', 'ultrafast',
        '-tune', 'zerolatency',  # 低延迟模式
        '-bufsize', '5000k',
        '-f', 'rtsp',
        '-rtsp_transport', 'tcp',
        rtmp_output_url
    ]
    pipe = subprocess.Popen(command, stdin=subprocess.PIPE)

    # 处理每一帧并显示结果
    while video_capture.cap.isOpened():
        ret, frame = video_capture.read()
        if not ret:
            continue

        frame = face_recognition.process_frame(frame, task_id)

        try:
            pipe.stdin.write(frame.tobytes())
        except BrokenPipeError:
            print("BrokenPipeError: Restarting FFmpeg process")
            pipe = subprocess.Popen(command, stdin=subprocess.PIPE)

        cv2.imshow("Face Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放资源
    video_capture.release()
    pipe.stdin.close()
    pipe.wait()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Detect and track unknown people using face recognition.')
    parser.add_argument('rtmp_input_url', type=str, help='RTMP input URL')
    parser.add_argument('rtmp_output_url', type=str, help='RTMP output URL')
    parser.add_argument('task_id', type=str, help='task_id')
    args = parser.parse_args()

    detect_and_track_unknown(args.rtmp_input_url, args.rtmp_output_url, args.task_id)
    # detect_and_track_unknown("rtsp://47.93.76.253:8554/camera_zkg", "rtsp://47.93.76.253:8554/mmm", 22)
