import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import csv
import time
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
    
    # [x1, y1, x2, y2] の形式で返す
    x1, y1 = int(x_min * w), int(y_min * h)
    x2, y2 = int(x_max * w), int(y_max * h)
    
    return [x1, y1, x2, y2]  # 修正されたバウンディングボックス形式


def landmarks_to_feature(landmarks):
    """骨格ランドマークを特徴量ベクトルに変換"""
    feature = []
    for lm in landmarks:
        feature.extend([lm.x, lm.y, lm.z])
    return np.array(feature, dtype=np.float32)

def average_csv(csv_paths, save_path):
    dfs = [pd.read_csv(p) for p in csv_paths]
    all_data = pd.concat(dfs)
    avg_df = all_data.groupby(['second', 'person_id']).mean().reset_index()
    avg_df.to_csv(save_path, index=False)
    print(f"平均化されたデータを {save_path} に保存しました。")

def calculate_iou(box1, box2):
    """2つの矩形のIOU(交差割合)を計算"""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)

    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area != 0 else 0

def record_pose(save_video_path, save_csv_path, camera_index=0, record_time=5):
    base_options = python.BaseOptions(model_asset_path='pose_landmarker_lite.task')
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_poses=2,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )
    detector = vision.PoseLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("カメラを開けませんでした。")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = 30

    out = cv2.VideoWriter(save_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    csv_file = open(save_csv_path, 'w', newline='')
    writer = csv.writer(csv_file)
    header = ["second", "person_id"] + [f"{c}{i}" for i in range(33) for c in ("x", "y", "z", "visibility")]
    writer.writerow(header)

    start_time = time.time()

    while time.time() - start_time < record_time:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp = int((time.time() - start_time) * 1000)
        result = detector.detect_for_video(mp_image, timestamp)

        detections = []
        feature_list = []
        landmarks_list = []

        if result.pose_landmarks:
            for pose_landmarks in result.pose_landmarks:
                bbox = landmarks_to_bbox(pose_landmarks, width, height)
                detections.append((bbox, 0.9))  # bboxとconfidenceをセットで渡す
                feature = landmarks_to_feature(pose_landmarks)
                feature_list.append(feature)
                landmarks_list.append(pose_landmarks)

        tracks = tracker.update_tracks(detections, frame=frame)

        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            color = get_color_for_id(track_id)

            track_bbox = track.to_ltwh()  # [x, y, w, h]
            best_landmarks = None
            best_iou = 0

            for det, lm in zip(detections, landmarks_list):
                bbox, _ = det  # bboxは [x1, y1, x2, y2]
                x1, y1, x2, y2 = bbox
                bbox_w = x2 - x1
                bbox_h = y2 - y1
                bbox_ltwh = [x1, y1, bbox_w, bbox_h]

                iou = calculate_iou(bbox_ltwh, track_bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_landmarks = lm

            if best_landmarks:
                landmarks_proto = landmark_pb2.NormalizedLandmarkList(
                    landmark=[
                        landmark_pb2.NormalizedLandmark(
                            x=lm.x,
                            y=lm.y,
                            z=lm.z,
                            visibility=lm.visibility
                        ) for lm in best_landmarks
                    ]
                )

                mp_drawing.draw_landmarks(
                    frame,
                    landmarks_proto,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=color, thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=color, thickness=2)
                )

                elapsed_time = time.time() - start_time
                second = int(elapsed_time)
                row = [second, track_id]
                for lm in best_landmarks:
                    row.extend([lm.x, lm.y, lm.z, lm.visibility])
                writer.writerow(row)

            x, y, w, h = map(int, track_bbox)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f'ID: {track_id}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        out.write(frame)
        cv2.imshow('Pose Tracking', frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    out.release()
    csv_file.close()
    cv2.destroyAllWindows()




def replay_ghost(original_video_path, avg_csv_path):
    cap = cv2.VideoCapture(original_video_path)
    avg_df = pd.read_csv(avg_csv_path)

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    sec_per_frame = 1 / fps

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 総フレーム数
    frame_index = 0  

    paused = False

    while cap.isOpened():
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break

        # 現在のフレーム番号に基づいて秒数を計算
        sec = int(frame_index / fps)

        subset = avg_df[avg_df["second"] == sec]
        if not subset.empty:
            for _, row in subset.iterrows():
                landmarks = []
                for i in range(33):
                    x = row[f'x{i}'] * frame_width
                    y = row[f'y{i}'] * frame_height
                    landmarks.append((int(x), int(y)))

                for connection in mp_pose.POSE_CONNECTIONS:
                    start_idx, end_idx = connection
                    cv2.line(frame, landmarks[start_idx], landmarks[end_idx], (0, 255, 0), 2)

        cv2.imshow('Ghost Replay', frame)

        key = cv2.waitKey(int(sec_per_frame * 1000)) & 0xFF
        if key == 27:  
            break
        elif key == ord(' '):  
            paused = not paused
        elif key == ord('f'):  
            frame_index = min(frame_index + 10, total_frames - 1)
        elif key == ord('b'):  
            frame_index = max(frame_index - 10, 0)

        if not paused:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)  # フレーム番号を設定
            frame_index += 1  

    cap.release()
    cv2.destroyAllWindows()




def main():
    print("モード選択: 1=カメラモード, 2=ビデオモード")
    mode = input("選択してください: ")

    if mode == '1':
        camera_index = int(input("使用するカメラ番号を入力してください(内蔵カメラ：０ or 外付けカメラ：１): "))
        n_samples = int(input("何回見本データを撮影しますか？: "))

        sample_csvs = []
        for i in range(1, n_samples + 1):
            video_path = f"data{i:02}.mp4"
            csv_path = f"data_csv{i:02}.csv"
            print(f"{i}回目の撮影開始...")
            record_pose(video_path, csv_path, camera_index=camera_index)
            sample_csvs.append(csv_path)

        average_csv_path = "average_sample.csv"
        average_csv(sample_csvs, average_csv_path)

        print("次に比較データを1回だけ撮影します。")
        compare_video = "compare.mp4"
        compare_csv = "compare_csv.csv"
        record_pose(compare_video, compare_csv, camera_index=camera_index)

        replay_choice = input("比較データの撮影が終了しました。ゴースト再生を始めますか？ (y/n): ")
        if replay_choice.lower() == 'y':
            print("ゴースト再生を開始します。")
            replay_ghost(compare_video, average_csv_path)
        
        replay_again='y'
        while replay_again !='n':
            replay_again = input("再生が終了しました。もう一度再生しますか？ (y/n): ")
            if replay_again.lower() == 'y':
                replay_ghost(compare_video, average_csv_path)
            


    elif mode == '2':
        print("ビデオモードはまだ未実装です。")

if __name__ == '__main__':
    main()
