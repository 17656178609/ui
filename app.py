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
model = YOLO("static/model/VOC.pt")  # 使用 GPU

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
behavior_stats = defaultdict(list)  # 用于存储行为检测统计

# 定义学习层次分类
# INVALID_LEARNING = ["Using_phone", "bow_head", "sleep", "bend", "turn_head"]
# SHALLOW_LEARNING = ["reading", "book", "writing"]
# DEEP_LEARNING = ["hand-raising", "Stand", "Discuss"]

INVALID_LEARNING = ["Th", "St"]
SHALLOW_LEARNING = ["Wr", "Re", "Lc"]
DEEP_LEARNING = ["Rh", "Gd", "Tg"]

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
    video_file = request.files.get("video")
    if video_file:
        # 保存视频文件
        video_path = os.path.join(UPLOAD_FOLDER, video_file.filename)
        video_file.save(video_path)
        # 将视频路径存储在会话中
        global current_video_path
        current_video_path = video_path

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
                # 视频播放结束，停止检测
                is_detecting = False
                cap.release()
                break
            # 进行检测
            results = model(frame)
            
            # 在原始帧上绘制检测结果
            annotated_frame = results[0].plot()
             # 统计检测到的行为
            for box in results[0].boxes:
                class_id = int(box.cls)
                class_name = model.names[class_id]
                behavior_stats[class_name].append(time.time())  # 记录检测时间
            # print(model.names)
            # 转换为JPEG格式
            ret, buffer = cv2.imencode('.jpg', annotated_frame)
            frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        cap.release()
        print("视频播放结束，检测已停止")
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
            # 进行检测
            results = model(frame)
            detection_boxes = results[0].boxes.xyxy.cpu().numpy().tolist()  # 获取检测框坐标

            # 统计检测到的行为
            for box in results[0].boxes:
                class_id = int(box.cls)
                class_name = model.names[class_id]
                behavior_stats[class_name].append(time.time())  # 记录检测时间
            # 将帧转换为 JPEG 格式
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            # 以流式响应返回帧
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


# 返回检测框数据
@app.route('/detection_data')
def detection_data():
    return jsonify(detection_boxes)

# 开始检测
@app.route("/start_detection")
def start_detection():
    global is_detecting
    is_detecting = True
    behavior_stats = defaultdict(list)  # 重置行为统计
    return "Detection started"

# 停止检测
@app.route("/stop_detection")
def stop_detection():
    global is_detecting
    is_detecting = False
    return redirect(url_for("detection_results"))  # 跳转到检测结果页面

# 检测结果页面
@app.route("/detection_results")
def detection_results():
    global behavior_stats

    # 计算每个二级指标的检测次数
    stats = {}
    for behavior in INVALID_LEARNING + SHALLOW_LEARNING + DEEP_LEARNING:
        stats[behavior] = len(behavior_stats.get(behavior, []))
    stats_json = json.dumps(stats)  # 将 stats 转换为 JSON 格式
    print(stats)
    return render_template("detection_results.html", stats_json=stats_json)

# 退出登录
@app.route("/logout")
def logout():
    return redirect(url_for("login"))

if __name__ == "__main__":
    app.run(debug=True)