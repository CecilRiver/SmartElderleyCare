import subprocess
import threading
from queue import Queue
from sql import insert_event_info
from qiniuConfig import upload_to_qiniu
from keras.preprocessing.image import img_to_array
import imutils
import cv2
from keras.models import load_model
import numpy as np
import os
import pandas as pd
from datetime import datetime
import argparse
import time

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
                print("Failed to read frame from source")
                break
            self.frame_queue.put(frame)

        self._reading = False

    def read(self):
        frame = self.frame_queue.get()
        return True, frame

    def get(self, prop_id):
        return self.cap.get(prop_id)

    def release(self):
        self._reading = False
        self.read_thread.join()
        self.cap.release()

def emotion_detection(rtmp_input_url, rtmp_output_url, task_id):
    # Parameters for loading data and models
    detection_model_path = 'haarcascade_files/haarcascade_frontalface_default.xml'
    emotion_model_path = 'models/_mini_XCEPTION.102-0.66.hdf5'

    # Loading models
    print("Loading models...")
    face_detection = cv2.CascadeClassifier(detection_model_path)
    emotion_classifier = load_model(emotion_model_path, compile=False)
    EMOTIONS = ["angry", "disgust", "scared", "happy", "sad", "surprised", "neutral"]

    # Sleep to ensure models are loaded properly
    time.sleep(2)

    # Video capture setup
    cv2.namedWindow('Input Stream')
    cv2.namedWindow('Emotion Detection')
    camera = MyVideoCapture(rtmp_input_url)
    fps = camera.get(cv2.CAP_PROP_FPS)

    if not camera.cap.isOpened():
        print("Error opening video stream or file")
        return

    # FFmpeg推流
    frame_width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
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
        '-f', 'rtsp',
        '-rtsp_transport', 'tcp',
        rtmp_output_url
    ]
    pipe = subprocess.Popen(command, stdin=subprocess.PIPE)

    # Main loop to process frames
    while camera.cap.isOpened():
        ret, frame = camera.read()

        if not ret:
            print("Failed to read frame")
            break

        # Display the input stream
        cv2.imshow('Input Stream', frame)

        # Resize frame for faster processing
        frame = imutils.resize(frame, width=300)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Face detection and emotion detection
        faces = face_detection.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30),
                                                flags=cv2.CASCADE_SCALE_IMAGE)
        canvas = np.zeros((250, 300, 3), dtype="uint8")
        frameClone = frame.copy()

        if len(faces) > 0:
            for (fX, fY, fW, fH) in faces:
                # Extract ROI for emotion detection
                roi_gray = gray[fY:fY + fH, fX:fX + fW]
                roi_gray = cv2.resize(roi_gray, (64, 64))
                roi_gray = roi_gray.astype("float") / 255.0
                roi_gray = img_to_array(roi_gray)
                roi_gray = np.expand_dims(roi_gray, axis=0)

                # Perform emotion detection
                preds = emotion_classifier.predict(roi_gray)[0]
                emotion_label = EMOTIONS[preds.argmax()]

                # Draw emotion label on the frame
                cv2.putText(frameClone, emotion_label, (fX, fY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
                cv2.rectangle(frameClone, (fX, fY), (fX + fW, fY + fH), (0, 0, 255), 2)

                # Save frame if emotion is "happy" and print related information
                if emotion_label == "happy":
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"happy_frame_{timestamp}.jpg"
                    cv2.imwrite(filename, frameClone)  # Save the frame
                    print(f"Detected 'happy' emotion. Frame saved as '{filename}'.")
                    image_url = upload_to_qiniu(filename)
                    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    event_desc = "happy"
                    insert_event_info(1, current_time, 'YF209', event_desc, 1, image_url, task_id)

                # Draw emotion probabilities on canvas
                for (i, (emotion, prob)) in enumerate(zip(EMOTIONS, preds)):
                    text = "{}: {:.2f}%".format(emotion, prob * 100)
                    w = int(prob * 300)
                    cv2.rectangle(canvas, (7, (i * 35) + 5), (w, (i * 35) + 35), (0, 0, 255), -1)
                    cv2.putText(canvas, text, (10, (i * 35) + 23), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)

        # Display frames with annotations
        cv2.imshow('Emotion Detection', frameClone)
        cv2.imshow("Emotion Probabilities", canvas)

        # Write frame to output stream
        try:
            # 修改前添加新行调整 frameClone 尺寸
            frameClone = cv2.resize(frameClone, (frame_width, frame_height))
            pipe.stdin.write(frameClone.tobytes())
        except BrokenPipeError:
            print("BrokenPipeError: Restarting FFmpeg process")
            pipe = subprocess.Popen(command, stdin=subprocess.PIPE)

        # Exit loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release video capture and writer, and close all windows
    camera.release()
    pipe.stdin.close()
    pipe.wait()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Emotion detection.')
    parser.add_argument('rtmp_input_url', type=str, help='RTMP input URL')
    parser.add_argument('rtmp_output_url', type=str, help='RTMP output URL')
    parser.add_argument('task_id', type=str, help='task_id')
    args = parser.parse_args()

    emotion_detection(args.rtmp_input_url, args.rtmp_output_url, args.task_id)
    #emotion_detection("rtsp://47.93.76.253:8554/camera_zkg", "rtsp://47.93.76.253:8554/mmm", 22)
