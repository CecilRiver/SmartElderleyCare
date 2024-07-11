import torch
import numpy as np
import os,cv2,time,torch,random,pytorchvideo,warnings,argparse,math
import Threading
import winsound
import time
import queue

freq = 2500
duration = 1000

from database_connector import DatabaseConnector
connector = DatabaseConnector(host='127.0.0.1', port=3306, database='traffic', user='root', password='2910347807')
warnings.filterwarnings("ignore",category=UserWarning)
os.environ["kmp_duplicate_lib_ok"]="true"
device = torch.device('cpu')
from pytorchvideo.transforms.functional import (
    uniform_temporal_subsample,
    short_side_scale_with_boxes,
    clip_boxes_to_image,)
from torchvision.transforms._functional_video import normalize
from pytorchvideo.data.ava import AvaLabeledVideoFramePaths
from pytorchvideo.models.hub import slowfast_r50_detection
from deep_sort.deep_sort import DeepSort

class FrameExtractor:
    cap_frame = None
    def __init__(self, stream_url, frame_interval):
        # 加载人脸检测器
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        # 创建视频捕获对象
        self.cap = cv2.VideoCapture(stream_url)

        # 检查视频捕获对象是否成功打开
        if not self.cap.isOpened():
            print("无法打开RTMP流")

        # 抽取帧的间隔
        self.frame_interval = frame_interval
        self.count = 1

    def read(self):
        # 逐帧读取视频流
        ret, frame = self.cap.read()

        if not ret:
            print("无法获取视频帧")

        # 每n帧抽取一帧
        if self.count == self.frame_interval:
            self.count = 1
            return True, frame
        else:
            self.count += 1
            return False, None

    def get(self, n):
        return self.cap.get(n)

    def release(self):
        # 释放资源
        self.cap.release()
class MyVideoCapture:
    
    def __init__(self, source):
        self.cap = cv2.VideoCapture(source)
        self.idx = -1
        self.end = False
        self.stack = []
        
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
        
def tensor_to_numpy(tensor):
    img = tensor.cpu().numpy().transpose((1, 2, 0))
    return img

def ava_inference_transform(
    clip, 
    boxes,
    num_frames = 32, #if using slowfast_r50_detection, change this to 32, 4 for slow 
    crop_size = 640, 
    data_mean = [0.45, 0.45, 0.45], 
    data_std = [0.225, 0.225, 0.225],
    slow_fast_alpha = 4, #if using slowfast_r50_detection, change this to 4, None for slow
):
    boxes = np.array(boxes)
    roi_boxes = boxes.copy()
    clip = uniform_temporal_subsample(clip, num_frames)
    clip = clip.float()
    clip = clip / 255.0
    height, width = clip.shape[2], clip.shape[3]
    boxes = clip_boxes_to_image(boxes, height, width)
    clip, boxes = short_side_scale_with_boxes(clip,size=crop_size,boxes=boxes,)
    clip = normalize(clip,
        np.array(data_mean, dtype=np.float32),
        np.array(data_std, dtype=np.float32),) 
    boxes = clip_boxes_to_image(boxes, clip.shape[2],  clip.shape[3])
    if slow_fast_alpha is not None:
        fast_pathway = clip
        slow_pathway = torch.index_select(clip,1,
            torch.linspace(0, clip.shape[1] - 1, clip.shape[1] // slow_fast_alpha).long())
        clip = [slow_pathway, fast_pathway]
    
    return clip, torch.from_numpy(boxes), roi_boxes

def plot_one_box(x, img, color=[100,100,100], text_info="None",
                 velocity=None, thickness=1, fontsize=0.5, fontthickness=1):
    # Plots one bounding box on image img
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness, lineType=cv2.LINE_AA)
    t_size = cv2.getTextSize(text_info, cv2.FONT_HERSHEY_TRIPLEX, fontsize , fontthickness+2)[0]
    cv2.rectangle(img, c1, (c1[0] + int(t_size[0]), c1[1] + int(t_size[1]*1.45)), color, -1)
    cv2.putText(img, text_info, (c1[0], c1[1]+t_size[1]+2), 
                cv2.FONT_HERSHEY_TRIPLEX, fontsize, [255,255,255], fontthickness)
    return img

def deepsort_update(Tracker, pred, xywh, np_img):
    outputs = Tracker.update(xywh, pred[:,4:5],pred[:,5].tolist(),cv2.cvtColor(np_img,cv2.COLOR_BGR2RGB))
    return outputs


def alarm(text):
    # 要检查的特定行为
    desired_actions = ["fall down", "smoke"]
    if any(action in text for action in desired_actions):
        print("报警：发现行为'{}'！".format(text))

import  cv2
import numpy as np

import  re
import datetime
def save_yolopreds_tovideo(yolo_preds, id_to_ava_labels, color_map,output_video):
    # for i, (im, pred) in enumerate(zip(yolo_preds.ims, yolo_preds.pred)):
    #     im=cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
    #     if pred.shape[0]:
    #         for j, (*box, cls, trackid, vx, vy) in enumerate(pred):
    #             if int(cls) != 0:
    #                 ava_label = ''
    #             elif trackid in id_to_ava_labels.keys():
    #                 ava_label = id_to_ava_labels[trackid].split(' ')[0]
    #
    #             else:
    #                 ava_label = 'Unknown'
    #                 #trackid:编号； yolo_preds.names[int(cls)]：yolo的识别结果； ava_label：行为预测
    #             if yolo_preds.names[int(cls)] == "person":
    #                 text = '{} {} {}'.format(int(trackid),yolo_preds.names[int(cls)],ava_label)
    #                 color = color_map[int(cls)]
    #                 im = plot_one_box(box,im,color,text)


        for i, (im, pred) in enumerate(zip(yolo_preds.ims, yolo_preds.pred)):
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            if pred.shape[0]:
                for j, (*box, cls, trackid, vx, vy) in enumerate(pred):
                    if int(cls) != 0:
                        ava_labels = []
                    elif trackid in id_to_ava_labels.keys():
                        ava_labels = id_to_ava_labels[trackid]

                    else:
                        ava_labels = ['Unknown']

                    if yolo_preds.names[int(cls)] == "person":
                        text = '{} {} {}'.format(int(trackid), yolo_preds.names[int(cls)], ', '.join(ava_labels))
                        color = color_map[int(cls)]
                        print(text)
                        im = plot_one_box(box, im, color, text)

                        alarm(text)


        im = im.astype(np.uint8)
        im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
        output_video.write(im)
        cv2.imshow("demo", im)


import cv2





def main(config):
    device = config.device
    imsize = config.imsize

    model = torch.hub.load('C:/Users/CecilRiver/.cache/torch/hub/ultralytics_yolov5_master', 'custom', path='C:/GitAlgorithm/yolo_slowfast/yolov5l6.pt', source='local').to(device)
    model.conf = config.conf
    model.iou = config.iou
    model.max_dex = 100

    if config.classes:
        model.classes = config.classes

    video_model = slowfast_r50_detection(True).eval().to(device)

    deepsort_tracker = DeepSort("deep_sort/deep_sort/deep/checkpoint/ckpt.t7")
    ava_labelnames,_=AvaLabeledVideoFramePaths.read_label_map("selfutils/temp.pbtxt")
    coco_color_map=[[random.randint(0,255)for _ in range(3)] for _ in range(80)]

    vide_save_path = config.output
    # video = FrameExtractor(config.input, 10)
    # width,height = int(video.get(3)),int(video.get(4))
    video = cv2.VideoCapture(config.input)
    width, height = int(video.get(3)), int(video.get(4))
    video.release()
    outputvideo = cv2.VideoWriter(vide_save_path,cv2.VideoWriter_fourcc(*'mp4v'), 25, (width,height))
    print("processing...")

    cap = MyVideoCapture(config.input)
    id_to_ava_labels = {}

    while not cap.end:
        ret, img =cap.read()
        #按下 'q' 键退出循环
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if not ret:
            continue
        #指定图像大小，yolo_preds包含YOLO模型对图片的预测结果
        yolo_preds=model([img],size=imsize)
        #deepsort实现目标跟踪
        deepsort_outputs = []
        for j in range(len(yolo_preds.pred)):
            temp = deepsort_update(deepsort_tracker, yolo_preds.pred[j].cpu(), yolo_preds.xywh[j][:, 0:4].cpu(),
                                   yolo_preds.ims[j])
            if len(temp) == 0:
                temp = np.ones((0, 8))
            deepsort_outputs.append(temp.astype(np.float32))

        yolo_preds.pred = deepsort_outputs
        if len(cap.stack) == 25:
            print(f"processing {cap.idx // 25}th second clips")
            clip = cap.get_video_clip()

            if yolo_preds.pred[0].shape[0]:
                inputs, inp_boxes, _ = ava_inference_transform(clip, yolo_preds.pred[0][:, 0:4], crop_size=imsize)
                inp_boxes = torch.cat([torch.zeros(inp_boxes.shape[0], 1), inp_boxes], dim=1)
                if isinstance(inputs, list):
                    inputs = [inp.unsqueeze(0).to(device) for inp in inputs]
                else:
                    inputs = inputs.unsqueeze(0).to(device)
                with torch.no_grad():
                    slowfaster_preds = video_model(inputs, inp_boxes.to(device))
                    slowfaster_preds = slowfaster_preds.cpu()

                for tid, avalabels in zip(yolo_preds.pred[0][:, 5].tolist(), slowfaster_preds.tolist()):
                    label_indices = sorted(range(len(avalabels)), key=lambda i: avalabels[i], reverse=True)[:3]
                    ava_labels = [ava_labelnames[idx + 1] for idx in label_indices]
                    probabilities = [avalabels[idx] for idx in label_indices]
                    labels_with_probabilities = [f"{label} ({probability:.2f})" for label, probability in
                                                 zip(ava_labels, probabilities)]
                    id_to_ava_labels[tid] = labels_with_probabilities

        save_yolopreds_tovideo(yolo_preds, id_to_ava_labels, coco_color_map,outputvideo)

    cap.release()
    outputvideo.release()
    connector.closeConnection()
    print('saved video to:', vide_save_path)


    

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default="0", help='test imgs folder or video or camera')
    parser.add_argument('--output', type=str, default="output7.14.mp4", help='folder to save result imgs, can not use input folder')
    # object detect config
    parser.add_argument('--imsize', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--iou', type=float, default=0.4, help='IOU threshold for NMS')
    parser.add_argument('--device', default='cuda', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--show', action='store_true', help='show img')
    config = parser.parse_args()
    
    if config.input.isdigit():
        print("using local camera.")
        config.input = int(config.input)

    main(config)


