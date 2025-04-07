import cv2
import mediapipe as mp
import numpy as np
import csv
import time
import pandas as pd
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
from deep_sort_realtime.deepsort_tracker import DeepSort

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

tracker = DeepSort(max_age=30, n_init=3, max_cosine_distance=0.3)

id_color_map = {}
def get_color_for_id(track_id):
    if track_id not in id_color_map:
        np.random.seed(int(track_id))
        id_color_map[track_id] = tuple(np.random.randint(100, 256, size=3).tolist())
    return id_color_map[track_id]

def landmarks_to_bbox(landmarks, w, h):
    x_vals = [lm.x for lm in landmarks]
    y_vals = [lm.y for lm in landmarks]
    x_min, x_max = min(x_vals), max(x_vals)
    y_min, y_max = min(y_vals), max(y_vals)
    x1, y1 = int(x_min * w), int(y_min * h)
    x2, y2 = int(x_max * w), int(y_max * h)
    return [x1, y1, x2 - x1, y2 - y1]

def split_csv_by_person_id(csv_path="multipose_landmarks.csv"):
    df = pd.read_csv(csv_path)
    unique_ids = df["person_id"].unique()
    for pid in unique_ids:
        person_df = df[df["person_id"] == pid]
        person_df.to_csv(f"person_{pid}_landmarks.csv", index=False)
    print("個別のCSVファイルに分割しました。")

def sort_csv_by_person_id(csv_path="multipose_landmarks.csv"):
    df = pd.read_csv(csv_path)
    df_sorted = df.sort_values(by=["person_id", "frame"])
    df_sorted.to_csv(csv_path, index=False)
    print("CSVをperson_idで並び替えました。")

def record_pose():
    camera_index = int(input("使用するカメラを選択してください（例:0=内蔵カメラ, 1=外部カメラ）: "))
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("カメラを開けませんでした。")
        return

    base_options = python.BaseOptions(model_asset_path='pose_landmarker_lite.task')
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_poses=2,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5)

    detector = vision.PoseLandmarker.create_from_options(options)

    ret, frame = cap.read()
    if not ret:
        print("カメラからフレームを取得できませんでした。")
        return

    frame = cv2.resize(frame, (320, 240))
    height, width = frame.shape[:2]
    fps = 20

    out_pose = cv2.VideoWriter('pose_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    out_original = cv2.VideoWriter('original_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    csv_file = open("multipose_landmarks.csv", mode='w', newline='')
    writer = csv.writer(csv_file)
    header = ["frame", "timestamp", "person_id"] + [f"{c}{i}" for i in range(33) for c in ("x", "y", "z", "visibility")]
    writer.writerow(header)

    frame_index = 0
    print("録画中: ESCキーで終了します。")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (320, 240))
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp = frame_index * int(1000 / fps)
        result = detector.detect_for_video(mp_image, timestamp)

        original = frame.copy()
        detections = []
        if result.pose_landmarks:
            for lm in result.pose_landmarks:
                box = landmarks_to_bbox(lm, width, height)
                detections.append(([box[0], box[1], box[2], box[3]], 1.0, 'person'))

        tracks = tracker.update_tracks(detections, frame=frame)

        if result.pose_landmarks:
            for idx, lm in enumerate(result.pose_landmarks):
                proto = landmark_pb2.NormalizedLandmarkList(landmark=[
                    landmark_pb2.NormalizedLandmark(x=p.x, y=p.y, z=p.z, visibility=p.visibility) for p in lm])

                mp_drawing.draw_landmarks(frame, proto, mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0,255,0), thickness=1, circle_radius=1),
                    mp_drawing.DrawingSpec(color=(0,0,255), thickness=1))

                row = [frame_index, time.time(), idx]
                for point in lm:
                    row.extend([point.x, point.y, point.z, point.visibility])
                writer.writerow(row)

        for tr in tracks:
            if not tr.is_confirmed(): continue
            tid = tr.track_id
            x1, y1, x2, y2 = map(int, tr.to_ltrb())
            color = get_color_for_id(tid)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f'ID:{tid}', (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        out_original.write(original)
        out_pose.write(frame)
        cv2.imshow('Pose Tracking (ESCで終了)', frame)

        if cv2.waitKey(20) & 0xFF == 27:
            break

        frame_index += 1

    csv_file.close()
    cap.release()
    out_pose.release()
    out_original.release()
    cv2.destroyAllWindows()

    split_csv_by_person_id()
    sort_csv_by_person_id()

    if input("録画終了。再生しますか？ (y/n): ").lower() == 'y':
        replay_pose_video()

def replay_pose_video():
    cap1 = cv2.VideoCapture('original_video.mp4')
    cap2 = cv2.VideoCapture('pose_video.mp4')

    frame_index = 0
    paused = False
    total_frames = int(min(cap1.get(cv2.CAP_PROP_FRAME_COUNT), cap2.get(cv2.CAP_PROP_FRAME_COUNT)))

    while cap1.isOpened() and cap2.isOpened():
        cap1.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        cap2.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

        r1, f1 = cap1.read()
        r2, f2 = cap2.read()
        if not r1 or not r2:
            break

        f1 = cv2.resize(f1, (640, 480))
        f2 = cv2.resize(f2, (640, 480))
        combo = np.hstack((f1, f2))
        cv2.putText(combo, f"Frame: {frame_index}/{total_frames}", (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow('Original | Pose', combo)

        key = cv2.waitKey(30) & 0xFF
        if key == 27:
            break
        elif key == ord(' '):
            paused = not paused
        elif key == ord('f'):
            frame_index = min(frame_index + 10, total_frames - 1)
        elif key == ord('b'):
            frame_index = max(frame_index - 10, 0)

        if not paused:
            frame_index += 1

    cap1.release()
    cap2.release()
    cv2.destroyAllWindows()

    replay_again = input("再生が終了しました。もう一度再生しますか？ (y/n): ")
    if replay_again.lower() == 'y':
        replay_pose_video()

if __name__ == '__main__':
    mode = input("モード選択: 1=カメラモード, 2=ビデオモード > ")
    if mode == '1':
        record_pose()
    elif mode == '2':
        print("ビデオモードは未実装です。")
    else:
        print("無効な入力です。")