from flask import Flask, render_template, request, redirect, url_for, Response
from ultralytics import YOLO
import os
import cv2
import threading
import json
from collections import defaultdict
import time
import torch
app = Flask(__name__)

# 模拟用户数据
users = {
    "admin": "password123",
    "user1": "123456"
}

# 加载训练好的 YOLOv8 模型
model = YOLO("best.pt") # 将模型加载到 GPU
# model = YOLO("yolov8n.pt")

if model is None:
    raise ValueError("模型加载失败，请检查模型文件路径。")
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
detection_stats = defaultdict(list)  # 用于存储时间维度的检测结果

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
    global detection_stats
    detection_stats = defaultdict(list)  # 重置统计结果

    video_file = request.files.get("video")
    if video_file:
        # 保存视频文件
        video_path = os.path.join(UPLOAD_FOLDER, video_file.filename)
        video_file.save(video_path)

        # 使用 YOLOv8 模型进行检测
        cap = cv2.VideoCapture(video_path)
        start_time = time.time()
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
            # 将帧转换为 GPU 张量
            # 调整输入图像的尺寸
            frame = cv2.resize(frame, (640, 640))  # 调整输入图像的尺寸
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # 转换为 RGB
            frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).float().to("cuda") / 255.0  # 归一化并加载到 GPU
            frame_tensor = frame_tensor.unsqueeze(0)  # 添加 batch 维度

            # 进行检测
            results = model(frame_tensor)
            current_time = time.time() - start_time
            for result in results:
                for box in result.boxes:
                    class_id = int(box.cls)
                    class_name = model.names[class_id]
                    detection_stats[class_name].append(current_time)  # 记录检测时间

        cap.release()
        return redirect(url_for("detection_results"))
    return "视频上传失败！"

# 实时视频流
def generate_frames():
    global camera, is_detecting, detection_stats
    camera = cv2.VideoCapture(0)  # 打开摄像头
    start_time = time.time()
    while is_detecting:
        success, frame = camera.read()
        if not success:
            break
        else:
            # 将帧转换为 GPU 张量
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # 转换为 RGB
            frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).float().to("cuda") / 255.0  # 归一化并加载到 GPU
            frame_tensor = frame_tensor.unsqueeze(0)  # 添加 batch 维度

            # 使用 YOLOv8 模型进行实时检测
            results = model(frame_tensor)
            current_time = time.time() - start_time
            annotated_frame = results[0].plot()  # 绘制检测结果

            # 更新统计结果
            for box in results[0].boxes:
                class_id = int(box.cls)
                class_name = model.names[class_id]
                detection_stats[class_name].append(current_time)  # 记录检测时间

            # 将帧转换为 JPEG 格式
            ret, buffer = cv2.imencode('.jpg', annotated_frame)
            frame = buffer.tobytes()

            # 以流式响应返回帧
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# 视频流路由
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# 开始检测
@app.route("/start_detection")
def start_detection():
    global is_detecting, detection_stats
    is_detecting = True
    detection_stats = defaultdict(list)  # 重置统计结果
    return "Detection started"

# 停止检测
@app.route("/stop_detection")
def stop_detection():
    global is_detecting, camera
    is_detecting = False
    if camera:
        camera.release()
    return redirect(url_for("detection_results"))

# 检测结果页面
@app.route("/detection_results")
def detection_results():
    # 将统计结果转换为 JSON 格式
    stats_json = json.dumps(detection_stats)
    return render_template("detection_results.html", stats_json=stats_json)

# 退出登录
@app.route("/logout")
def logout():
    return redirect(url_for("login"))

if __name__ == "__main__":
    app.run(debug=True)