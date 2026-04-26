from flask import Flask, render_template, request
from ultralytics import YOLO, RTDETR
from flask import send_from_directory
import os
from collections import defaultdict
from sklearn.cluster import KMeans
import numpy as np
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



def extract_jersey_hue(frame, x1, y1, x2, y2):
    h_box = y2 - y1
    w_box = x2 - x1

    ys = y1 + int(h_box * 0.25)
    ye = y1 + int(h_box * 0.65)
    xs = x1 + int(w_box * 0.15)
    xe = x1 + int(w_box * 0.85)

    ys = max(0, ys); ye = min(frame.shape[0], ye)
    xs = max(0, xs); xe = min(frame.shape[1], xe)

    roi = frame[ys:ye, xs:xe]
    if roi.size < 50:
        return None

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv,
                       np.array([0, 50, 50]),
                       np.array([180, 255, 255]))

    hues = hsv[:, :, 0][mask > 0]
    if len(hues) < 30:
        return None

    hist, _ = np.histogram(hues, bins=18, range=(0, 180))
    return int(np.argmax(hist)) * 10


def classify_teams(player_hues_dict):
    ids = [pid for pid, hues in player_hues_dict.items() if len(hues) >= 5]
    if len(ids) < 2:
        return {}

    X = np.array([[np.mean(player_hues_dict[i])] for i in ids])
    km = KMeans(n_clusters=2, n_init=10, random_state=0)
    labels = km.fit_predict(X)

    return {pid: int(lbl) for pid, lbl in zip(ids, labels)}

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

    player_hues = defaultdict(list)
    player_teams = {}

    TEAM_LOCKED = False
    INIT_FRAMES = 100
    frame_num = 0

    if is_image:
        results = model(input_path, conf=0.5)[0]

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
        frame_num += 1
        if not ret:
            break

        if model_name == "model_RTDETR.pt":
            results = model(frame, conf=0.5)[0]

            annotated_frame = results.plot()

            boxes = results.boxes

            if boxes is not None:
                for box in boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])

                    confidences.append(conf)
                    label = model.names[cls]

                    if label == "ball":
                        ball_detected = True
        else:
            results = model.track(
                frame,
                tracker="bytetrack.yaml",
                persist=True,
                conf=0.5,
                verbose=False)[0]

            annotated_frame = frame.copy()
            

            # récupérer les detections
            boxes = results.boxes

            if boxes is not None:
                current_players = 0
                for box in boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    

                    confidences.append(conf)
                    label = model.names[cls]

                    if label == "human":
                        if box.id is not None:
                            tid = int(box.id[0])
                        else:
                            continue

                        hue = extract_jersey_hue(frame, x1, y1, x2, y2)

                        if hue is not None:
                            player_hues[tid].append(hue)
                            player_hues[tid] = player_hues[tid][-50:]

                        # LOCK équipes après quelques frames
                        if not TEAM_LOCKED:
                            if frame_num > INIT_FRAMES and len(player_hues) >= 4:
                                player_teams = classify_teams(player_hues)
                                TEAM_LOCKED = True

                        # assignation équipe
                        if TEAM_LOCKED and tid in player_teams:
                            team = player_teams[tid]
                        else:
                            team = -1

                        # couleurs
                        if team == 0:
                            color = (255, 0, 0)   # équipe A
                            label_text = f"ID {tid} A"
                        elif team == 1:
                            color = (0, 0, 255)   # équipe B
                            label_text = f"ID {tid} B"
                        else:
                            color = (150, 150, 150)
                            label_text = f"ID {tid} ?"

                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 4)
                        cv2.putText(annotated_frame, label_text,
                                    (x1, y1 - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5, (255,255,255), 2)

                    if label == "ball":
                            ball_detected = True

                            # dessiner le ballon
                            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0,165,255), 3)
                            cv2.putText(annotated_frame, "BALL",
                                        (x1, y1 - 5),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        0.5, (255,255,255), 2)
                            
                    if label == "rim":
                            ball_detected = True

                            # dessiner le ballon
                            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (165,165,255), 2)
                            cv2.putText(annotated_frame, "RIM",
                                        (x1, y1 - 5),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        0.5, (255,255,255), 3)

        out.write(annotated_frame)

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
    print("SERVEUR LANCÉ")
    app.run(debug=True)
