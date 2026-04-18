from flask import Flask, render_template, request
from ultralytics import YOLO, RTDETR
from flask import send_from_directory
import os
import cv2

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

    filename = file.filename.lower()

    is_image = filename.endswith((".jpg", ".jpeg", ".png"))
    is_video = filename.endswith((".mp4", ".avi", ".mov"))

    model_path = os.path.join("models", model_name)

    if model_name == "model_RTDETR.pt":
        model = RTDETR(model_path)
    else:
        model = YOLO(model_path)

    if is_image:
        results = model(input_path)[0]

        annotated = results.plot()

        output_path = os.path.join("static", "output.jpg")
        cv2.imwrite(output_path, annotated)

        boxes = results.boxes

        max_players = 0
        ball_detected = False
        confidences = []

        if boxes is not None:
            for box in boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])

                confidences.append(conf)
                label = model.names[cls]

                if label == "human":
                    max_players += 1

                if label == "ball":
                    ball_detected = True

        avg_conf = (sum(confidences) / len(confidences) * 100) if confidences else 0

        return render_template(
            "result.html",
            file_type="image",
            file_name="output.jpg",
            players=max_players,
            confidence=round(avg_conf, 1),
            ball="Oui" if ball_detected else "Non",
            model_name=model_name
        )

    # ouvrir vidéo
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

    model.predictor = None 
    all_track_ids = set()
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if model_name == "model_RTDETR.pt":
            results = model(frame)[0]
        else: 
            results = model.track(
                frame,
                tracker="bytetrack.yaml",
                persist=True,
                conf=0.7,
                verbose=False)[0]

        annotated_frame = results.plot()
        out.write(annotated_frame)

        # récupérer les detections
        boxes = results.boxes

        if boxes is not None:
            current_players = 0
            for box in boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])

                confidences.append(conf)
                label = model.names[cls]

                if label == "human":
                    if box.id is not None:
                        all_track_ids.add(int(box.id[0]))

                if label == "ball":
                    ball_detected = True

    max_players = len(all_track_ids)

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
        file_name="output.mp4",
        file_type="video",
        video_name=video_name,
        players=max_players,
        confidence=round(avg_conf, 1),
        ball=ball_text,
        model_name=model_name
    )

if __name__ == "__main__":
    app.run(debug=True)
