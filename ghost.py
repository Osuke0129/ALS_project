import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import csv
import time
import os
import shutil
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
from deep_sort_realtime.deepsort_tracker import DeepSort
from dtaidistance import dtw_ndim

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
tracker = DeepSort(max_age=60, n_init=5, max_cosine_distance=0.3)
id_color_map = {}

def get_color_for_id(track_id):
    if track_id not in id_color_map:
        np.random.seed(int(track_id))
        id_color_map[track_id] = tuple(np.random.randint(100, 256, size=3).tolist())
    return id_color_map[track_id]

def landmarks_to_bbox(landmarks, w, h):
    x_vals = [lm.x for lm in landmarks]
    y_vals = [lm.y for lm in landmarks]
    x1, y1 = int(min(x_vals) * w), int(min(y_vals) * h)
    x2, y2 = int(max(x_vals) * w), int(max(y_vals) * h)
    return [x1, y1, x2, y2]

def calculate_iou(box1, box2):
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

def normalize_landmarks(df):
    x_cols = [f"x{i}" for i in range(33)]
    y_cols = [f"y{i}" for i in range(33)]
    z_cols = [f"z{i}" for i in range(33)]

    # 欠損値がある行を除去
    df = df.dropna(subset=x_cols + y_cols + z_cols).reset_index(drop=True)

    # X座標の正規化
    x_min, x_max = df[x_cols].min().min(), df[x_cols].max().max()
    if x_max != x_min:
        df[x_cols] = (df[x_cols] - x_min) / (x_max - x_min)
    else:
        print("⚠ x_max == x_min のため正規化スキップ")

    # Y座標の正規化
    y_min, y_max = df[y_cols].min().min(), df[y_cols].max().max()
    if y_max != y_min:
        df[y_cols] = (df[y_cols] - y_min) / (y_max - y_min)
    else:
        print("⚠ y_max == y_min のため正規化スキップ")

    return df


def dtw_align(reference, target):
    ref_values = reference.drop(columns=['frame', 'person_id']).values
    tar_values = target.drop(columns=['frame', 'person_id']).values
    path = dtw_ndim.warping_path(ref_values, tar_values)
    aligned_target = np.array([tar_values[j] for i, j in path])
    aligned_df = pd.DataFrame(aligned_target, columns=reference.drop(columns=['frame', 'person_id']).columns)
    aligned_df.insert(0, 'person_id', reference['person_id'].iloc[0])
    aligned_df.insert(0, 'frame', np.arange(len(aligned_target)))
    return aligned_df

def average_csv_dtw(csv_paths, save_path):
    dfs = [normalize_landmarks(pd.read_csv(p)) for p in csv_paths]
    reference = dfs[0]
    aligned_dfs = [reference]
    for df in dfs[1:]:
        aligned_dfs.append(dtw_align(reference, df))
    all_data = pd.concat(aligned_dfs)
    avg_df = all_data.groupby(['frame', 'person_id']).mean().reset_index()
    avg_df.to_csv(save_path, index=False)
    print(f"平均化データを {save_path} に保存しました。")

def replay_ghost(avg_csv_path, compare_csv_path, compare_video_path, save_video_path=None):
    cap = cv2.VideoCapture(compare_video_path)

    # データ読み込み＆正規化
    avg_df_raw = pd.read_csv(avg_csv_path)
    compare_df_raw = pd.read_csv(compare_csv_path)
    avg_df = normalize_landmarks(avg_df_raw)
    compare_df = normalize_landmarks(compare_df_raw)

    # DTW整列（平均を比較者に合わせる）
    aligned_avg = dtw_align(compare_df, avg_df)

    # 動画情報取得
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or np.isnan(fps): fps = 15
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    sec_per_frame = 1 / fps
    total_frames = len(aligned_avg)
    frame_index = 0
    paused = False

    # 保存設定
    out = None
    if save_video_path:
        out = cv2.VideoWriter(save_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    while True:
        if not paused:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, frame = cap.read()
            if not ret: break
        frame_copy = frame.copy()

        # 平均（緑）の骨格のみ描画
        if frame_index < len(aligned_avg):
            row = aligned_avg.iloc[frame_index]
            landmarks = [(int(row[f"x{i}"] * frame_width), int(row[f"y{i}"] * frame_height)) for i in range(33)]
            for start, end in mp_pose.POSE_CONNECTIONS:
                cv2.line(frame_copy, landmarks[start], landmarks[end], (0, 255, 0), 2)

        cv2.imshow('Ghost Replay (Average Only)', frame_copy)
        if out:
            out.write(frame_copy)

        key = cv2.waitKey(int(sec_per_frame * 1000)) & 0xFF
        if key == 27: break
        elif key == ord(' '): paused = not paused
        elif key == ord('f'): frame_index = min(total_frames - 1, frame_index + 5); paused = True
        elif key == ord('b'): frame_index = max(0, frame_index - 5); paused = True

        if not paused:
            frame_index += 1
            if frame_index >= total_frames:
                break

    cap.release()
    if out: out.release()
    cv2.destroyAllWindows()





def record_pose(video_path, csv_path, camera_index=0, record_time=5):
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

    width, height, fps = int(cap.get(3)), int(cap.get(4)), 15
    out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    # 元動画保存用
    original_video_path = video_path.replace('_video.mp4', '_original.mp4')
    original_out = cv2.VideoWriter(original_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    csv_file = open(csv_path, 'w', newline='')
    writer = csv.writer(csv_file)
    writer.writerow(["frame", "person_id"] + [f"{c}{i}" for i in range(33) for c in ("x", "y", "z", "visibility")])

    start_time = time.time()
    frame_count = 0

    while time.time() - start_time < record_time:
        ret, frame = cap.read()
        if not ret:
            break

        original_out.write(frame)  # 元動画に保存

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = detector.detect_for_video(mp_image, int((time.time() - start_time) * 1000))

        detections = []
        landmarks_list = []

        if result.pose_landmarks:
            for lm in result.pose_landmarks:
                x_vals = [pt.x for pt in lm]
                y_vals = [pt.y for pt in lm]
                x1 = int(min(x_vals) * width)
                y1 = int(min(y_vals) * height)
                x2 = int(max(x_vals) * width)
                y2 = int(max(y_vals) * height)
                w, h = x2 - x1, y2 - y1

                if w <= 0 or h <= 0:
                    continue

                detections.append(([x1, y1, w, h], 0.9))
                landmarks_list.append(lm)

        tracks = tracker.update_tracks(detections, frame=frame)

        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            color = get_color_for_id(track_id)
            track_bbox = track.to_ltwh()
            best_landmarks, best_iou = None, 0

            for det, lm in zip(detections, landmarks_list):
                x, y, w, h = det[0]
                iou = calculate_iou([x, y, w, h], track_bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_landmarks = lm

            if best_landmarks:
                writer.writerow(
                    [frame_count, track_id] +
                    [v for pt in best_landmarks for v in (pt.x, pt.y, pt.z, pt.visibility)]
                )

                landmarks_proto = landmark_pb2.NormalizedLandmarkList(
                    landmark=[
                        landmark_pb2.NormalizedLandmark(
                            x=pt.x, y=pt.y, z=pt.z, visibility=pt.visibility
                        ) for pt in best_landmarks
                    ]
                )
                mp_drawing.draw_landmarks(
                    frame,
                    landmarks_proto,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=color, thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=color, thickness=2)
                )

            x, y, w, h = map(int, track_bbox)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f'ID: {track_id}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        out.write(frame)
        cv2.imshow('Pose Tracking', frame)

        frame_count += 1
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    out.release()
    original_out.release()  # 忘れずに
    csv_file.close()
    cv2.destroyAllWindows()
    print(f"録画されたフレーム数: {frame_count}")


def main():
    base_training = "C:/Users/217ki/ALS_project/training_data"
    base_play = "C:/Users/217ki/ALS_project/play_data"
    base_result = "C:/Users/217ki/ALS_project/result"

    while True:
        print("\nモード選択: 1=録画, 2=検証, q=終了")
        mode = input("選択してください: ")
        if mode == 'q':
            break

        timestamp = time.strftime("%Y%m%d_%H%M%S")

        if mode == '1':
            rec_user_name = input("訓練者の名前を入力してください:")
            folder_name = f"{rec_user_name}_{timestamp}"
            folder_path = os.path.join(base_training, folder_name)
            os.makedirs(folder_path, exist_ok=True)
            n_samples = int(input("何回見本データを撮影しますか？: "))
            camera_index = int(input("カメラ番号 (0:内蔵, 1:外付け): "))
            record_time = int(input("録画時間（秒）: "))
            sample_csvs = []

            for i in range(1, n_samples + 1):
                if input(f"{i}回目の撮影開始？(y/n): ").lower() != 'y':
                    continue
                sub_folder = os.path.join(folder_path, f"{rec_user_name}_{i:02}")
                os.makedirs(sub_folder, exist_ok=True)
                t = time.strftime("%Y%m%d_%H%M%S")
                video_path = os.path.join(sub_folder, f"{rec_user_name}_{t}_video.mp4")
                csv_path = os.path.join(sub_folder, f"{rec_user_name}_{t}_csv.csv")
                record_pose(video_path, csv_path, camera_index, record_time)
                sample_csvs.append(csv_path)

            avg_csv_path = os.path.join(folder_path, f"{rec_user_name}_{timestamp}_average.csv")
            average_csv_dtw(sample_csvs, avg_csv_path)

        elif mode == '2':
            comp_user_name = input("比較者の名前を入力してください:")
            folder_name = f"{comp_user_name}_{timestamp}"
            folder_path_play = os.path.join(base_play, folder_name)
            os.makedirs(folder_path_play, exist_ok=True)

            camera_index = int(input("カメラ番号 (0:内蔵, 1:外付け): "))
            record_time = int(input("録画時間（秒）: "))

            # 最新の訓練データから平均CSV取得
            folders = sorted(
                [f for f in os.listdir(base_training) if os.path.isdir(os.path.join(base_training, f))],
                reverse=True
            )
            if not folders:
                print("訓練データが存在しません。")
                continue

            selected_folder = os.path.join(base_training, folders[0])
            avg_csv_list = [f for f in os.listdir(selected_folder) if f.endswith("_average.csv")]
            if not avg_csv_list:
                print("平均CSVが見つかりません。")
                continue

            avg_csv_path = os.path.join(selected_folder, avg_csv_list[0])

            # 比較者の動画とCSVを撮影
            compare_video = os.path.join(folder_path_play, f"{comp_user_name}_{timestamp}_compare.mp4")
            compare_csv = os.path.join(folder_path_play, f"{comp_user_name}_{timestamp}_compare.csv")
            if input("比較データの撮影開始？(y/n): ").lower() == 'y':
                record_pose(compare_video, compare_csv, camera_index, record_time)

            # 結果保存フォルダ作成
            result_folder = os.path.join(base_result, folder_name)
            os.makedirs(result_folder, exist_ok=True)
            ghost_video_path = os.path.join(result_folder, f"{comp_user_name}_{timestamp}_ghost.mp4")

            # 比較動画の再生（平均CSV・比較CSVのDTW整列付き）
            if input("比較動画を再生しますか？(y/n): ").lower() == 'y':
                replay_ghost(avg_csv_path, compare_csv, compare_video_path=compare_video, save_video_path=ghost_video_path)
                while input("もう一度再生しますか？(y/n): ").lower() == 'y':
                    replay_ghost(avg_csv_path, compare_csv, compare_video_path=compare_video, save_video_path=None)


if __name__ == '__main__':
    main()
