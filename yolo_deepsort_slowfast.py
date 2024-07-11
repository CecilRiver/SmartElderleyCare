import random
import shlex
import subprocess
import sys

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
    device = "cuda"
    imsize = 640
    model = torch.hub.load('C:/Users/CecilRiver/.cache/torch/hub/ultralytics_yolov5_master', 'custom',
                           path='C:/GitAlgorithm/yolo_slowfast/yolov5l6.pt', source='local').to(device)
    model.conf = 0.4
    model.iou = 0.4
    model.max_dex = 100
    deepsort_tracker = DeepSort("C:/GitAlgorithm/yolo_slowfast/deep_sort/deep_sort/deep/checkpoint/ckpt.t7")
    coco_color_map = [[random.randint(0, 255) for _ in range(3)] for _ in range(80)]
    ava_labelnames,_=AvaLabeledVideoFramePaths.read_label_map("C:/GitAlgorithm/yolo_slowfast/selfutils/temp.pbtxt")
    id_to_ava_labels = {}
    video_model = slowfast_r50_detection(True).eval().to(device)

    def __init__(self, ds, up):
        print("processing----------->")
        self.ds = ds
        self.up = up

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

    def plot_one_box(slef, x, img, color=[100, 100, 100], text_info="None",
                     velocity=None, thickness=1, fontsize=0.5, fontthickness=1):

        c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
        cv2.rectangle(img, c1, c2, color, thickness, lineType=cv2.LINE_AA)
        t_size = cv2.getTextSize(text_info, cv2.FONT_HERSHEY_TRIPLEX, fontsize, fontthickness + 2)[0]
        cv2.rectangle(img, c1, (c1[0] + int(t_size[0]), c1[1] + int(t_size[1] * 1.45)), color, -1)
        cv2.putText(img, text_info, (c1[0], c1[1] + t_size[1] + 2),
                    cv2.FONT_HERSHEY_TRIPLEX, fontsize, [255, 255, 255], fontthickness)
        return img

    def alarm(text):
        # 要检查的特定行为
        desired_actions = ["fall down","smoke"]
        if any(action in text for action in desired_actions):
            print("报警：发现行为'{}'！".format(text))
            # 修改此行为提交报警信息
            #g.get_json_response(
                #f'{conf.MANAGE_SERVER_BASE}/task/alert/set/?&task_id={id}&alert=1')
        #else:
            #g.get_json_response(
                #f'{conf.MANAGE_SERVER_BASE}/task/alert/set/?&task_id={id}&alert=0')
    # 返回处理后的帧图片，和该图片中的person数量
    def process(self, input):
        cap = MyVideoCapture(self.ds)

        # command = ['ffmpeg',
        #            '-y', '-an',  # 无需询问即可覆盖输出文件
        #            '-f', 'rawvideo',  # 强制输入或输出文件格式
        #            '-vcodec', 'rawvideo',  # 设置视频编解码器。这是-codec:v的别名
        #            '-pix_fmt', 'bgr24',  # 设置像素格式
        #            '-s', f'{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}',
        #            # 设置图像大小
        #            '-r', str(int(cap.get(cv2.CAP_PROP_FPS))),  # 设置帧率
        #            '-i', '-',  # 输入
        #            '-c:v', 'libx264',  # 编解码器
        #            '-pix_fmt', 'yuv420p',  # 像素格式
        #            '-preset', 'ultrafast',  # 调节编码速度和质量的平衡
        #            '-f', 'flv',  # 强制输入或输出文件格式
        #            '-tune', 'zerolatency',  # 视频类型和视觉优化
        #            '-force_key_frames', '1',
        #            '-g', f'{str(int(cap.get(cv2.CAP_PROP_FPS)))}',
        #            self.up]

        command = ['ffmpeg',
                   '-y', '-an',  # 无需询问即可覆盖输出文件
                   '-f', 'rawvideo',  # 强制输入或输出文件格式
                   '-vcodec', 'rawvideo',  # 设置视频编解码器。这是-codec:v的别名
                   '-pix_fmt', 'bgr24',  # 设置像素格式
                   '-s', f'{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}',
                   # 设置图像大小
                   '-r', str(int(cap.get(cv2.CAP_PROP_FPS))),  # 设置帧率
                   '-i', '-',  # 输入，从stdin读取视频帧
                   '-c:v', 'libx264',  # 编解码器
                   '-pix_fmt', 'yuv420p',  # 像素格式
                   '-preset', 'ultrafast',  # 调节编码速度和质量的平衡
                   '-tune', 'zerolatency',  # 视频类型和视觉优化
                   '-f', 'rawvideo',  # 输出为原始视频
                   '-']  # 输出到stdout

        pipe = subprocess.Popen(shlex.join(command), shell=False, stdin=subprocess.PIPE)
        # 指定图像大小，yolo_preds包含YOLO模型对图片的预测结果
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
                temp = self.deepsort_update(self.deepsort_tracker, yolo_preds.pred[j].cpu(), yolo_preds.xywh[j][:, 0:4].cpu(),
                                            yolo_preds.ims[j])
                if len(temp) == 0:
                    temp = np.ones((0, 8))
                deepsort_outputs.append(temp.astype(np.float32))

            yolo_preds.pred = deepsort_outputs
            if len(cap.stack) == 25:

                print(f"processing {cap.idx // 25}th second clips")
                clip = cap.get_video_clip()
                if yolo_preds.pred[0].shape[0]:
                    inputs, inp_boxes, _ = self.ava_inference_transform(clip, yolo_preds.pred[0][:, 0:4], crop_size=self.imsize)
                    inp_boxes = torch.cat([torch.zeros(inp_boxes.shape[0], 1), inp_boxes], dim=1)
                    if isinstance(inputs, list):
                        inputs = [inp.unsqueeze(0).to(self.device) for inp in inputs]
                    else:
                        inputs = inputs.unsqueeze(0).to(self.device)
                    with torch.no_grad():
                        slowfaster_preds = self.video_model(inputs, inp_boxes.to(self.device))
                        slowfaster_preds = slowfaster_preds.cpu()
                    for tid, avalabels in zip(yolo_preds.pred[0][:, 5].tolist(), slowfaster_preds.tolist()):
                        label_indices = sorted(range(len(avalabels)), key=lambda i: avalabels[i], reverse=True)[:3]
                        ava_labels = [self.ava_labelnames[idx + 1] for idx in label_indices]
                        probabilities = [avalabels[idx] for idx in label_indices]
                        labels_with_probabilities = [f"{label} ({probability:.2f})" for label, probability in
                                                     zip(ava_labels, probabilities)]
                        self.id_to_ava_labels[tid] = labels_with_probabilities

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

                        if yolo_preds.names[int(cls)] == "person":
                            text = '{} {} {}'.format(int(trackid), yolo_preds.names[int(cls)], ', '.join(ava_labels))
                            color = self.coco_color_map[int(cls)]
                            im = self.plot_one_box(box, im, color, text)

                           # self.alarm(text)

                im = im.astype(np.uint8)
                im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
                cv2.imshow("demo", im)
                # 按下 'q' 键退出循环
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                try:
                    pipe.stdin.write(im.tobytes())
                except OSError as e:
                    print('推流问题')


if __name__ == "__main__":
    ds = 0
    print(ds)
    up = 1
    print(up)
    yolo_deepsort_slowfast = yolo_deepsort_slowfast(ds, up)
    yolo_deepsort_slowfast.process(ds)







