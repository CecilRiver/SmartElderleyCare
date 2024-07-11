import random
import subprocess
import shlex
import sys

import cv2
import numpy as np
import torch
from deep_sort.deep_sort import DeepSort



class yolo_deepsort:
    
    device = "cuda"
    imsize = 640
    model = torch.hub.load('C:/Users/CecilRiver/.cache/torch/hub/ultralytics_yolov5_master', 'custom',
                           path='C:/GitAlgorithm/yolo_slowfast/yolov5l6.pt', source='local').to(device)
    model.conf = 0.4
    model.iou = 0.4
    model.max_dex = 100
    deepsort_tracker = DeepSort("C:\GitAlgorithm\yolo_slowfast\deep_sort\deep_sort\deep\checkpoint\ckpt.t7")
    coco_color_map = [[random.randint(0, 255) for _ in range(3)] for _ in range(80)]

    def __init__(self):
        print("processing----------->")


    def deepsort_update(self,Tracker, pred, xywh, np_img):
        outputs = Tracker.update(xywh, pred[:,4:5],pred[:,5].tolist(),cv2.cvtColor(np_img,cv2.COLOR_BGR2RGB))
        return outputs

    def plot_one_box(slef,x, img, color=[100,100,100], text_info="None",
                     velocity=None, thickness=1, fontsize=0.5, fontthickness=1):

        c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
        cv2.rectangle(img, c1, c2, color, thickness, lineType=cv2.LINE_AA)
        t_size = cv2.getTextSize(text_info, cv2.FONT_HERSHEY_TRIPLEX, fontsize , fontthickness+2)[0]
        cv2.rectangle(img, c1, (c1[0] + int(t_size[0]), c1[1] + int(t_size[1]*1.45)), color, -1)
        cv2.putText(img, text_info, (c1[0], c1[1]+t_size[1]+2),
                    cv2.FONT_HERSHEY_TRIPLEX, fontsize, [255,255,255], fontthickness)
        return img


    #返回处理后的帧图片，和该图片中的person数量
    def process(self,frame1):
        # 指定图像大小，yolo_preds包含YOLO模型对图片的预测结果
        yolo_preds = self.model([frame1], size=self.imsize)
        # deepsort实现目标跟踪
        deepsort_outputs = []
        # 记录 "person" 类别的数量
        person_count = 0
        for j in range(len(yolo_preds.pred)):
            temp = self.deepsort_update(self.deepsort_tracker, yolo_preds.pred[j].cpu(), yolo_preds.xywh[j][:, 0:4].cpu(),
                                   yolo_preds.ims[j])
            if len(temp) == 0:
                temp = np.ones((0, 8))
            deepsort_outputs.append(temp.astype(np.float32))
        yolo_preds.pred = deepsort_outputs
        for i, (im, pred) in enumerate(zip(yolo_preds.ims, yolo_preds.pred)):
            im=cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
            if pred.shape[0]:
                for j, (*box, cls, trackid, vx, vy) in enumerate(pred):
                    text = '{} {} '.format(int(trackid), yolo_preds.names[int(cls)])
                    color = self.coco_color_map[int(cls)]
                    im = self.plot_one_box(box,im,color,text)
                    # 计数 "person" 类别的出现次数
                    if yolo_preds.names[int(cls)] == "person":
                        person_count += 1
            im = im.astype(np.uint8)
            # 给下方中间画框
            text_box_width = 200
            text_box_height = 30
            text_box_x = (im.shape[1] - text_box_width) // 2
            text_box_y = im.shape[0] - text_box_height - 10
            cv2.rectangle(im, (text_box_x, text_box_y), (text_box_x + text_box_width, text_box_y + text_box_height),
                          (255, 255, 255), -1)

            #计算拥挤度
            crowd = person_count / 10.0

            #显示拥挤度文本
            text = f"Crowd Rate: {crowd}"
            text_color = (255, 0, 0)  # 红色字体
            font_scale = 0.8
            font_thickness = 1
            (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
            text_x = text_box_x + (text_box_width - text_width) // 2
            text_y = text_box_y + (text_box_height + text_height) // 2
            cv2.putText(im, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, font_thickness,
                        cv2.LINE_AA)

            # 将图像从RGB转换为BGR
            im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)

        return im,person_count

def get_pipe(self):
    command = ['ffmpeg',
               '-y', '-an',                                         # 无需询问即可覆盖输出文件
               '-f', 'rawvideo',                                        # 强制输入或输出文件格式
               '-vcodec', 'rawvideo',                               # 设置视频编解码器。这是-codec:v的别名
               '-pix_fmt', 'bgr24',                                 # 设置像素格式
               '-s', f'{self.frame_width}x{self.frame_height}',     # 设置图像大小
               '-r', str(self.fps),                                 # 设置帧率
               '-i', '-',                                           # 输入
               '-c:v', 'libx264',                                   # 编解码器
               '-pix_fmt', 'yuv420p',                               # 像素格式
               '-preset', 'ultrafast',                              # 调节编码速度和质量的平衡
               '-f', 'flv',                                         # 强制输入或输出文件格式
               '-tune', 'zerolatency',                                  # 视频类型和视觉优化
               '-force_key_frames', '1',
               '-g', f'{self.fps}',
               self.output_rtmp]
    return subprocess.Popen(shlex.join(command), shell=False, stdin=subprocess.PIPE)

if __name__=="__main__":
    # 读取图像
    # frame1 = cv2.imread("C:/Users/CecilRiver/Desktop/people/Images/IMG_20200322_103408.jpg")
    # frame2 = cv2.imread("C:/Users/CecilRiver/Desktop/people/Images/IMG_20200322_135301(1).jpg")
    # frame3 = cv2.imread("C:/Users/CecilRiver/Desktop/people/Images/IMG_20200322_135301.jpg")
    # yolo_deepsort = yolo_deepsort()
    # # 创建自定义窗口
    # cv2.namedWindow("demo", cv2.WINDOW_NORMAL)
    # im,person_count = yolo_deepsort.process(frame1)
    # print(person_count)
    # cv2.imshow("demo", im)
    # cv2.waitKey(0)
    # im,person_count = yolo_deepsort.process(frame2)
    # print(person_count)
    # cv2.imshow("demo", im)
    # cv2.waitKey(0)
    # im,person_count = yolo_deepsort.process(frame3)
    # print(person_count)
    # cv2.imshow("demo", im)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # 读取视频
    #video_path = "C:/Users/CecilRiver/ScreenRecord/20230714_153414.mp4"
    ds = sys.argv[1]
    print(ds)
    up = sys.argv[2]
    print(up)
    cap = cv2.VideoCapture(ds)
    command = ['ffmpeg',
               '-y', '-an',  # 无需询问即可覆盖输出文件
               '-f', 'rawvideo',  # 强制输入或输出文件格式
               '-vcodec', 'rawvideo',  # 设置视频编解码器。这是-codec:v的别名
               '-pix_fmt', 'bgr24',  # 设置像素格式
               '-s', f'{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}',  # 设置图像大小
               '-r', str(int(cap.get(cv2.CAP_PROP_FPS))),  # 设置帧率
               '-i', '-',  # 输入
               '-c:v', 'libx264',  # 编解码器
               '-pix_fmt', 'yuv420p',  # 像素格式
               '-preset', 'ultrafast',  # 调节编码速度和质量的平衡
               '-f', 'flv',  # 强制输入或输出文件格式
               '-tune', 'zerolatency',  # 视频类型和视觉优化
               '-force_key_frames', '1',
               '-g', f'{str(int(cap.get(cv2.CAP_PROP_FPS)))}',
               up]
    pipe= subprocess.Popen(shlex.join(command), shell=False, stdin=subprocess.PIPE)



    # 创建自定义窗口
    #cv2.namedWindow("demo", cv2.WINDOW_NORMAL)

    yolo_deepsort = yolo_deepsort()
    while True:
        # 读取视频帧
        ret, frame = cap.read()
        if not ret:
            break

        # 处理帧
        im, person_count = yolo_deepsort.process(frame)
        #print(person_count)

        # 显示帧
        #cv2.imshow("demo", im)

        # 按下 'q' 键退出循环
        #if cv2.waitKey(1) & 0xFF == ord('q'):
        #    break

        try:
            pipe.stdin.write(im.tobytes())
        except OSError as e:
            print('推流问题')

    # 释放资源
    cap.release()
    cv2.destroyAllWindows()
