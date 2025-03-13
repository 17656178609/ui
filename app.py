from flask import Flask, render_template, request, redirect, url_for, Response, jsonify
from ultralytics import YOLO
import os
import cv2
import threading
import json
from collections import defaultdict
import time
import torch
import numpy as np

app = Flask(__name__)

# 模拟用户数据
users = {
    "admin": "password123",
    "user1": "123456"
}

# 加载训练好的 YOLOv8 模型
model = YOLO("yolov8n.pt")  # 使用 GPU

# 上传视频保存路径
UPLOAD_FOLDER = "uploads"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# 检测结果保存路径
RESULTS_FOLDER = "static/results"
if not os.path.exists(RESULTS_FOLDER):
    os.makedirs(RESULTS_FOLDER)

# 全局变量用于实时视频流
camera = None
is_detecting = False
detection_boxes = []  # 用于存储检测框的坐标
current_video_path = None # 用于存储当前视频路径
# 登录页面
@app.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        if username in users and users[username] == password:
            return redirect(url_for("behavior_detection"))
        else:
            return "用户名或密码错误！"
    return render_template("login.html")

# 行为检测页面
@app.route("/behavior_detection", methods=["GET", "POST"])
def behavior_detection():
    return render_template("behavior_detection.html")

# 处理上传的视频
@app.route("/upload_video", methods=["POST"])
def upload_video():
    global detection_boxes
    detection_boxes = []  # 重置检测框数据

    video_file = request.files.get("video")
    if video_file:
        # 保存视频文件
        video_path = os.path.join(UPLOAD_FOLDER, video_file.filename)
        # video_file.save(video_path)
        # 将视频路径存储在会话中
        global current_video_path
        current_video_path = video_path
        # # 使用 YOLOv8 模型进行检测
        # cap = cv2.VideoCapture(video_path)
        # start_time = time.time()
        # while cap.isOpened():
        #     success, frame = cap.read()
        #     if not success:
        #         break

        #     # 调整输入图像的尺寸
        #     frame = cv2.resize(frame, (640, 640))  # 调整输入图像的尺寸

            # # 将帧转换为 GPU 张量
            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # 转换为 RGB
            # frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).float().to("cuda") / 255.0  # 归一化并加载到 GPU
            # frame_tensor = frame_tensor.unsqueeze(0)  # 添加 batch 维度

        #     # 进行检测
        #     results = model(frame_tensor)
        #     detection_boxes = results[0].boxes.xyxy.cpu().numpy().tolist()  # 获取检测框坐标

        # cap.release()
        return redirect(url_for("behavior_detection"))
    return "视频上传失败！"


# 添加视频检测流路由
@app.route('/video_detection_feed')
def video_detection_feed():
    def generate():
        global current_video_path, is_detecting
        if not current_video_path:
            return
            
        cap = cv2.VideoCapture(current_video_path)
        while is_detecting and cap.isOpened():
            success, frame = cap.read()
            if not success:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # 循环播放
                continue
            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # 转换为 RGB
            # frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).float().to("cuda") / 255.0  # 归一化并加载到 GPU
            # frame_tensor = frame_tensor.unsqueeze(0)  # 添加 batch 维度
            # 进行检测
            results = model(frame)
            
            # 在原始帧上绘制检测结果
            annotated_frame = results[0].plot()
            
            # 转换为JPEG格式
            ret, buffer = cv2.imencode('.jpg', annotated_frame)
            frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        cap.release()

    return Response(generate(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')
# 实时视频流
def generate_frames():
    global camera, is_detecting, detection_boxes
    camera = cv2.VideoCapture(0)  # 打开摄像头
    start_time = time.time()
    while is_detecting:
        success, frame = camera.read()
        if not success:
            break
        else:
            # 调整输入图像的尺寸
            frame = cv2.resize(frame, (640, 640))  # 调整输入图像的尺寸

            # 将帧转换为 GPU 张量
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # 转换为 RGB
            frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).float().to("cuda") / 255.0  # 归一化并加载到 GPU
            frame_tensor = frame_tensor.unsqueeze(0)  # 添加 batch 维度

            # 进行检测
            results = model(frame_tensor)
            detection_boxes = results[0].boxes.xyxy.cpu().numpy().tolist()  # 获取检测框坐标

            # 将帧转换为 JPEG 格式
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            # 以流式响应返回帧
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# 视频流路由
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# 返回检测框数据
@app.route('/detection_data')
def detection_data():
    return jsonify(detection_boxes)

# 开始检测
@app.route("/start_detection")
def start_detection():
    global is_detecting
    is_detecting = True
    return "Detection started"

# 停止检测
@app.route("/stop_detection")
def stop_detection():
    global is_detecting
    is_detecting = False
    return "Detection stopped"

# 检测结果页面
@app.route("/detection_results")
def detection_results():
    return render_template("detection_results.html")

# 退出登录
@app.route("/logout")
def logout():
    return redirect(url_for("login"))

if __name__ == "__main__":
    app.run(debug=True)