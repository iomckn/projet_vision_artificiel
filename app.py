from flask import Flask, render_template, request
from ultralytics import YOLO
from flask import send_from_directory
import os
import cv2
import glob
import shutil

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "runs/detect/predict"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/")
def index():
    return render_template("index.html")



def convert_to_mp4(input_path, output_path):
    cap = cv2.VideoCapture(input_path)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)

    cap.release()
    out.release()

@app.route('/video/<filename>')
def video(filename):
    return send_from_directory('runs/detect/predict', filename)

@app.route("/upload", methods=["POST"])

def upload():

    file = request.files["video"]
    model_name = request.form["model"]

    input_path = os.path.join("uploads", file.filename)
    file.save(input_path)

    model = YOLO(os.path.join("models", model_name))

    # 🔥 ouvrir vidéo
    cap = cv2.VideoCapture(input_path)

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 🔥 sortie DIRECT MP4
    output_path = os.path.join("static", "output.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    max_players = 0
    ball_detected = False
    confidences = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)[0]

        annotated_frame = results.plot()
        out.write(annotated_frame)

        # 🔥 récupérer les detections
        boxes = results.boxes

        if boxes is not None:
            current_players = 0
            for box in boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])

                confidences.append(conf)
                label = model.names[cls]

                if label == "human":
                    current_players += 1

                if label == "ball":
                    ball_detected = True

            max_players = max(max_players, current_players)

    # moyenne confiance
    if len(confidences) > 0:
        avg_conf = sum(confidences) / len(confidences) * 100
    else:
        avg_conf = 0

    ball_text = "Oui" if ball_detected else "Non"
    cap.release()
    out.release()

    video_name = "output.mp4"

    return render_template(
        "result.html",
        video_name=video_name,
        players=max_players,
        confidence=round(avg_conf, 1),
        ball=ball_text,
        model_name=model_name
    )

if __name__ == "__main__":
    app.run(debug=True)