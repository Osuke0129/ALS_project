import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import csv
import time

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def record_pose():
    base_options = python.BaseOptions(model_asset_path='pose_landmarker_full.task')
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_poses=5,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5)

    detector = vision.PoseLandmarker.create_from_options(options)

    camera_index = int(input("使用するカメラを選択してください（例:0=内蔵カメラ, 1=外部カメラ）: "))
    cap = cv2.VideoCapture(camera_index)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 30
    width, height = 480, 480

    out_pose = cv2.VideoWriter('pose_video.mp4', fourcc, fps, (width, height))
    out_original = cv2.VideoWriter('original_video.mp4', fourcc, fps, (width, height))

    start_time = int(cv2.getTickCount())
    last_timestamp = 0

    csv_file = open("multipose_landmarks.csv", mode='w', newline='')
    csv_writer = csv.writer(csv_file)
    header = ["timestamp", "person_idx"] + [item for j in range(33) for item in (f"x{j}", f"y{j}", f"z{j}", f"visibility{j}")]
    csv_writer.writerow(header)

    print("録画中: ESCキー=終了")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_resized = cv2.resize(frame, (width, height))
        rgb_frame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        current_time = int(cv2.getTickCount())
        timestamp = int((current_time - start_time) / cv2.getTickFrequency() * 1000)

        if timestamp <= last_timestamp:
            timestamp = last_timestamp + 1
        last_timestamp = timestamp

        detection_result = detector.detect_for_video(mp_image, timestamp)

        original_frame = frame_resized.copy()

        if detection_result.pose_landmarks:
            for person_idx, pose_landmarks in enumerate(detection_result.pose_landmarks):
                landmark_proto = landmark_pb2.NormalizedLandmarkList(
                    landmark=[
                        landmark_pb2.NormalizedLandmark(
                            x=lm.x,
                            y=lm.y,
                            z=lm.z,
                            visibility=lm.visibility
                        ) for lm in pose_landmarks
                    ]
                )

                mp_drawing.draw_landmarks(
                    frame_resized,
                    landmark_proto,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=3),
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(0,0,255), thickness=2))

                h, w, _ = frame_resized.shape
                row = [time.time(), person_idx]
                for idx, landmark in enumerate(pose_landmarks):
                    x, y = int(landmark.x * w), int(landmark.y * h)
                    cv2.putText(frame_resized, f"({x},{y})", (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                                0.4, (255, 255, 255), 1, cv2.LINE_AA)
                    row.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])

                csv_writer.writerow(row)

                center_landmark = pose_landmarks[0]
                center_x, center_y = int(center_landmark.x * w), int(center_landmark.y * h)
                person_label = f"No{person_idx+1}"
                cv2.putText(frame_resized, person_label, (center_x, center_y - 20), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 255, 255), 2, cv2.LINE_AA)

        out_original.write(original_frame)
        out_pose.write(frame_resized)

        cv2.imshow('Multipose Detection', frame_resized)

        if cv2.waitKey(5) & 0xFF == 27:
            break

    csv_file.close()
    cap.release()
    out_pose.release()
    out_original.release()
    cv2.destroyAllWindows()

    replay = input("録画が終了しました。録画した映像を再生しますか？ (y/n): ")
    if replay.lower() == 'y':
        playback_pose()

def playback_pose():
    cap_original = cv2.VideoCapture('original_video.mp4')
    cap_pose = cv2.VideoCapture('pose_video.mp4')

    fps = 30
    frame_index = 0
    paused = False

    total_frames = int(cap_original.get(cv2.CAP_PROP_FRAME_COUNT))

    while cap_original.isOpened() and cap_pose.isOpened():
        cap_original.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        cap_pose.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

        ret1, original_frame = cap_original.read()
        ret2, pose_frame = cap_pose.read()

        if not ret1 or not ret2:
            break

        combined_frame = np.hstack((original_frame, pose_frame))
        cv2.imshow('Original (Left) vs Pose Detection (Right)', combined_frame)

        key = cv2.waitKey(33) & 0xFF
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

    cap_original.release()
    cap_pose.release()
    cv2.destroyAllWindows()

    replay_again = input("再生が終了しました。もう一度再生しますか？ (y/n): ")
    if replay_again.lower() == 'y':
        playback_pose()

def video_mode():
    video_path = input("動画ファイル名を入力してください（.mp4形式）: ")
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("動画ファイルを開けませんでした。ファイル名を確認してください。")
        return

    base_options = python.BaseOptions(model_asset_path='pose_landmarker_full.task')
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_poses=5,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5)

    detector = vision.PoseLandmarker.create_from_options(options)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_pose = cv2.VideoWriter('pose_video.mp4', fourcc, fps, (width, height))
    out_original = cv2.VideoWriter('original_video.mp4', fourcc, fps, (width, height))

    csv_file = open("multipose_landmarks.csv", mode='w', newline='')
    csv_writer = csv.writer(csv_file)
    header = ["timestamp", "person_idx"] + [item for j in range(33) for item in (f"x{j}", f"y{j}", f"z{j}", f"visibility{j}")]
    csv_writer.writerow(header)

    frame_index = 0
    start_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_resized = cv2.resize(frame, (width, height))
        rgb_frame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        timestamp = int((time.time() - start_time) * 1000)

        detection_result = detector.detect_for_video(mp_image, timestamp)

        original_frame = frame_resized.copy()

        if detection_result.pose_landmarks:
            for person_idx, pose_landmarks in enumerate(detection_result.pose_landmarks):
                landmark_proto = landmark_pb2.NormalizedLandmarkList(
                    landmark=[
                        landmark_pb2.NormalizedLandmark(
                            x=lm.x,
                            y=lm.y,
                            z=lm.z,
                            visibility=lm.visibility
                        ) for lm in pose_landmarks
                    ]
                )

                mp_drawing.draw_landmarks(
                    frame_resized,
                    landmark_proto,
                    mp_pose.POSE_CONNECTIONS)

                h, w, _ = frame_resized.shape
                row = [time.time(), person_idx]
                for idx, landmark in enumerate(pose_landmarks):
                    x, y = int(landmark.x * w), int(landmark.y * h)
                    cv2.putText(frame_resized, f"({x},{y})", (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                                0.4, (255, 255, 255), 1, cv2.LINE_AA)
                    row.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])

                csv_writer.writerow(row)

        out_original.write(original_frame)
        out_pose.write(frame_resized)

        frame_index += 1

    csv_file.close()
    cap.release()
    out_pose.release()
    out_original.release()
    cv2.destroyAllWindows()

    replay = input("骨格抽出が終了しました。再生しますか？ (y/n): ")
    if replay.lower() == 'y':
        playback_pose()



if __name__ == '__main__':
    mode = input("モード選択: 1=カメラモード, 2=ビデオモード > ")

    if mode == '1':
        record_pose()
    elif mode == '2':
        video_mode()
    else:
        print("無効な入力です。");