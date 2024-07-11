# 跌倒检测
# done

import random
import subprocess
from datetime import datetime

from sql import insert_event_info
from qiniuConfig import upload_to_qiniu

import cv2
import numpy as np
import torch
from pytorchvideo.models.hub import slowfast_r50_detection

from deep_sort.deep_sort import DeepSort
from pytorchvideo.data.ava import AvaLabeledVideoFramePaths
from pytorchvideo.transforms.functional import (
    uniform_temporal_subsample,
    short_side_scale_with_boxes,
    clip_boxes_to_image,)
from torchvision.transforms._functional_video import normalize
import argparse

class MyVideoCapture:
    def __init__(self, source):
        self.cap = cv2.VideoCapture(source)
        self.idx = -1
        self.end = False
        self.stack = []

    def get(self, prop_id):
        if self.cap.isOpened():
            return self.cap.get(prop_id)

    def read(self):
        self.idx += 1
        ret, img = self.cap.read()
        if ret:
            self.stack.append(img)
        else:
            self.end = True
        return ret, img

    def to_tensor(self, img):
        img = torch.from_numpy(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        return img.unsqueeze(0)

    def get_video_clip(self):
        assert len(self.stack) > 0, "clip length must larger than 0 !"
        self.stack = [self.to_tensor(img) for img in self.stack]
        clip = torch.cat(self.stack).permute(-1, 0, 1, 2)

        del self.stack
        self.stack = []
        return clip

    def release(self):
        self.cap.release()

class yolo_deepsort_slowfast:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    imsize = 640
    model = torch.hub.load('C:/Users/CecilRiver/.cache/torch/hub/ultralytics_yolov5_master', 'custom',
                           path='C:/GitAlgorithm/yolo_slowfast/yolov5l6.pt', source='local').to(device)
    model.conf = 0.2  # 调低置信度阈值
    model.iou = 0.4
    model.max_dex = 100
    deepsort_tracker = DeepSort("C:/GitAlgorithm/yolo_slowfast/deep_sort/deep_sort/deep/checkpoint/ckpt.t7")
    coco_color_map = [[random.randint(0, 255) for _ in range(3)] for _ in range(80)]
    ava_labelnames, _ = AvaLabeledVideoFramePaths.read_label_map("C:/GitAlgorithm/yolo_slowfast/selfutils/temp.pbtxt")
    id_to_ava_labels = {}
    video_model = slowfast_r50_detection(True).eval().to(device)

    def __init__(self, ds, up, rtmp_output_url, task_id):
        print("processing----------->")
        self.ds = ds
        self.up = up
        self.task_id = task_id
        self.rtmp_output_url = rtmp_output_url

    @staticmethod
    def ava_inference_transform(
            clip,
            boxes,
            num_frames=32,  # if using slowfast_r50_detection, change this to 32, 4 for slow
            crop_size=640,
            data_mean=[0.45, 0.45, 0.45],
            data_std=[0.225, 0.225, 0.225],
            slow_fast_alpha=4,  # if using slowfast_r50_detection, change this to 4, None for slow
    ):
        boxes = np.array(boxes)
        roi_boxes = boxes.copy()
        clip = uniform_temporal_subsample(clip, num_frames)
        clip = clip.float()
        clip = clip / 255.0
        height, width = clip.shape[2], clip.shape[3]
        boxes = clip_boxes_to_image(boxes, height, width)
        clip, boxes = short_side_scale_with_boxes(clip, size=crop_size, boxes=boxes, )
        clip = normalize(clip,
                         np.array(data_mean, dtype=np.float32),
                         np.array(data_std, dtype=np.float32), )
        boxes = clip_boxes_to_image(boxes, clip.shape[2], clip.shape[3])
        if slow_fast_alpha is not None:
            fast_pathway = clip
            slow_pathway = torch.index_select(clip, 1,
                                              torch.linspace(0, clip.shape[1] - 1,
                                                             clip.shape[1] // slow_fast_alpha).long())
            clip = [slow_pathway, fast_pathway]

        return clip, torch.from_numpy(boxes), roi_boxes

    def deepsort_update(self, Tracker, pred, xywh, np_img):
        outputs = Tracker.update(xywh, pred[:, 4:5], pred[:, 5].tolist(), cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB))
        return outputs

    def plot_one_box(self, x, img, color=[100, 100, 100], text_info="None",
                     velocity=None, thickness=1, fontsize=0.5, fontthickness=1, text_color=[255, 255, 255]):

        c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
        cv2.rectangle(img, c1, c2, color, thickness, lineType=cv2.LINE_AA)
        t_size = cv2.getTextSize(text_info, cv2.FONT_HERSHEY_TRIPLEX, fontsize, fontthickness + 2)[0]
        cv2.rectangle(img, c1, (c1[0] + int(t_size[0]), c1[1] + int(t_size[1] * 1.45)), color, -1)
        cv2.putText(img, text_info, (c1[0], c1[1] + t_size[1] + 2),
                    cv2.FONT_HERSHEY_TRIPLEX, fontsize, text_color, fontthickness)
        return img

    def alarm(self, text):
        # 要检查的特定行为
        desired_actions = ["fall down"]
        if any(action in text for action in desired_actions):
            print("报警：发现行为'{}'！".format(text))

    def process(self, input):
        cap = MyVideoCapture(self.ds)

        # frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        # frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # fps = int(cap.get(cv2.CAP_PROP_FPS))

        # FFmpeg推流
        ffmpeg_path = "ffmpeg"  # FFmpeg可执行文件路径
        command = [
            ffmpeg_path,
            '-y', '-an',
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-pix_fmt', 'bgr24',
            '-s', "640x360",
            '-r', "10",
            '-i', '-',
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-preset', 'ultrafast',
            '-f', 'rtsp',
            '-rtsp_transport', 'tcp',
            self.rtmp_output_url
        ]
        pipe = subprocess.Popen(command, stdin=subprocess.PIPE)

        while not cap.end:
            ret, img = cap.read()
            # 按下 'q' 键退出循环
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            if not ret:
                continue
            # 指定图像大小，yolo_preds包含YOLO模型对图片的预测结果
            yolo_preds = self.model([img], size=self.imsize)

            # deepsort实现目标跟踪
            deepsort_outputs = []

            for j in range(len(yolo_preds.pred)):
                temp = self.deepsort_update(self.deepsort_tracker, yolo_preds.pred[j].cpu(),
                                            yolo_preds.xywh[j][:, 0:4].cpu(),
                                            yolo_preds.ims[j])
                if len(temp) == 0:
                    temp = np.ones((0, 8))
                deepsort_outputs.append(temp.astype(np.float32))

            yolo_preds.pred = deepsort_outputs
            if len(cap.stack) == 25:
                print(f"processing {cap.idx // 25}th second clips")
                clip = cap.get_video_clip()
                if yolo_preds.pred[0].shape[0]:
                    inputs, inp_boxes, _ = self.ava_inference_transform(clip, yolo_preds.pred[0][:, 0:4],
                                                                        crop_size=self.imsize)
                    inp_boxes = torch.cat([torch.zeros(inp_boxes.shape[0], 1), inp_boxes], dim=1)
                    if isinstance(inputs, list):
                        inputs = [inp.unsqueeze(0).to(self.device) for inp in inputs]
                    else:
                        inputs = inputs.unsqueeze(0).to(self.device)
                    with torch.no_grad():
                        slowfaster_preds = self.video_model(inputs, inp_boxes.to(self.device))
                        slowfaster_preds = slowfaster_preds.cpu()
                    for tid, avalabels in zip(yolo_preds.pred[0][:, 5].tolist(), slowfaster_preds.tolist()):
                        label_indices = sorted(range(len(avalabels)), key=lambda i: avalabels[i], reverse=True)

                        # 增加"fall down"的权重
                        for i, idx in enumerate(label_indices):
                            if self.ava_labelnames[idx + 1] == "fall down":
                                print(f"Original fall down probability: {avalabels[idx]}")
                                avalabels[idx] *= 333
                                print(f"Adjusted fall down probability: {avalabels[idx]}")

                        # 只保留 fall down 标签
                        fall_down_index = None
                        for i, idx in enumerate(label_indices):
                            if self.ava_labelnames[idx + 1] == "fall down":
                                fall_down_index = idx
                                break

                        # 如果找到了 fall down 标签，保存其概率
                        if fall_down_index is not None:
                            ava_labels = [self.ava_labelnames[fall_down_index + 1]]
                            probabilities = [avalabels[fall_down_index]]
                            labels_with_probabilities = [f"{label} ({probability:.2f})" for label, probability in
                                                         zip(ava_labels, probabilities)]
                            self.id_to_ava_labels[tid] = labels_with_probabilities
                        else:
                            self.id_to_ava_labels[tid] = ["fall down (0.00)"]  # 如果没有找到 fall down 标签，设置默认值

            for i, (im, pred) in enumerate(zip(yolo_preds.ims, yolo_preds.pred)):
                im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                if pred.shape[0]:
                    for j, (*box, cls, trackid, vx, vy) in enumerate(pred):
                        if int(cls) != 0:
                            ava_labels = []
                        elif trackid in self.id_to_ava_labels.keys():
                            ava_labels = self.id_to_ava_labels[trackid]
                        else:
                            ava_labels = ['Unknown']

                        if yolo_preds.names[int(cls)] == "person" and "fall down" in ava_labels[0]:
                            text = '{} {} {}'.format(int(trackid), yolo_preds.names[int(cls)], ', '.join(ava_labels))
                            color = [0, 0, 255]  # Red color for bounding box
                            text_color = [0, 255, 0]  # Green color for text
                            im = self.plot_one_box(box, im, color, text, text_color=text_color)
                            # self.alarm(text)

                            # 保存fall down概率大于1的图片
                            fall_down_prob = float(ava_labels[0].split('(')[1].strip(')'))

                            if fall_down_prob > 1:
                                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                filename = f"fall_down_frame_{timestamp}.jpg"
                                cv2.imwrite(filename, im)  # Save the frame
                                print(f"Detected Fall Down. Frame saved as '{filename}'.")
                                image_url = upload_to_qiniu(filename)
                                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                event_desc = f"Fall Down !!!"
                                insert_event_info(5, current_time, 'YF209', event_desc, 0, image_url, self.task_id)



                im = im.astype(np.uint8)
                im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
                try:
                    pipe.stdin.write(im.tobytes())
                except BrokenPipeError:
                    print("BrokenPipeError: Restarting FFmpeg process")
                    pipe = subprocess.Popen(command, stdin=subprocess.PIPE)

        cap.release()
        pipe.stdin.close()
        pipe.wait()
        print("Processing completed.")

if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description='Fall down detection using YOLOv5, DeepSORT, and SlowFast.')
    # parser.add_argument('rtmp_input_url', type=str, help='RTMP input URL')
    # parser.add_argument('rtmp_output_url', type=str, help='RTMP output URL')
    # parser.add_argument('task_id', type=str, help='task_id')
    # args = parser.parse_args()
    #
    # yolo_deepsort_slowfast = yolo_deepsort_slowfast(args.rtmp_input_url, up=1, rtmp_output_url=args.rtmp_output_url, task_id=args.task_id)
    # yolo_deepsort_slowfast.process(args.rtmp_input_url)
    yolo_deepsort_slowfast = yolo_deepsort_slowfast("C://User//CecilRiver//ScreenRecord//跌倒视频", up=1, rtmp_output_url="rtsp://47.93.76.253:8554/mmm",
                                                    task_id=22)
    yolo_deepsort_slowfast.process("C://User//CecilRiver//ScreenRecord//跌倒视频")