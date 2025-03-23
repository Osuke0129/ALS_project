import cv2
import mediapipe as mp
import csv
import time
import numpy as np

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# 録画モード（CSVファイルと動画を作成）
def record_pose():
    cap = cv2.VideoCapture(0)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 30.0
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_pose = cv2.VideoWriter('recorded_pose.mp4', fourcc, fps, (int(cap.get(3)), int(cap.get(4))))
    out_original = cv2.VideoWriter('moto_pose.mp4', fourcc, fps, (int(cap.get(3)), int(cap.get(4))))

    csv_file = open('landmarks.csv', 'w', newline='')
    csv_writer = csv.writer(csv_file)

    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    print("録画中: ESCキー=終了")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        original_frame = frame.copy()  # 骨格を描かない元フレームを保存

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)

        landmarks_frame = []
        if results.pose_landmarks:
            landmarks_frame.append(time.time())
            for landmark in results.pose_landmarks.landmark:
                landmarks_frame.extend([landmark.x, landmark.y])

        csv_writer.writerow(landmarks_frame)

        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        h, w, _ = frame.shape
        if results.pose_landmarks:
            for idx, landmark in enumerate(results.pose_landmarks.landmark):
                x, y = int(landmark.x * w), int(landmark.y * h)
                cv2.putText(frame, f"{idx}: ({x}, {y})", (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, (255, 255, 255), 1, cv2.LINE_AA)

        
        out_original.write(original_frame)  # 元動画（骨格なし）
        out_pose.write(frame)               # 骨格描画あり動画

        cv2.imshow('Recording Pose', frame)
        if cv2.waitKey(5) & 0xFF == 27:
            break

    # 必ず両方をreleaseする
    out_original.release()
    out_pose.release()
    cap.release()
    cv2.destroyAllWindows()

    replay = input("録画が終了しました。録画した映像を再生しますか？ (y/n): ")
    if replay.lower() == 'y':
        playback_pose()

# 再生モード（CSVファイルと動画を同期再生）
def playback_pose():
    cap_original = cv2.VideoCapture('moto_pose.mp4')
    cap_pose = cv2.VideoCapture('recorded_pose.mp4')

    data = []
    with open('landmarks.csv', 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            data_row = [float(x) for x in row]
            data.append(data_row)

    paused = False
    frame_index = 0
    total_frames = len(data)

    print("再生中: スペース=一時停止/再開, f=早送り(10フレーム), b=巻き戻し(10フレーム), ESC=終了")

    while frame_index < total_frames:
        cap_original.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        cap_pose.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

        ret1, frame_original = cap_original.read()
        ret2, frame_pose = cap_pose.read()

        if not ret1 or not ret2:
            break

        h, w, _ = frame_pose.shape

        landmarks = data[frame_index][1:]  # 最初の要素はタイムスタンプ
        for i in range(0, len(landmarks), 2):
            x = int(landmarks[i] * w)
            y = int(landmarks[i + 1] * h)
            cv2.circle(frame_pose, (x, y), 3, (0, 255, 0), -1)
            cv2.putText(frame_pose, f"{i//2}: ({x}, {y})", (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, (255, 255, 255), 1, cv2.LINE_AA)
        
        result=np.hstack((frame_original,frame_pose))
        cv2.imshow('moto | Pose PlayBack', result)

        key = cv2.waitKey(int(1000 / 30)) & 0xFF

        if key == 27:
            break
        elif key == ord(' '):
            paused = not paused
        elif key == ord('f'):
            frame_index = min(frame_index + 10, total_frames - 1)
        elif key == ord('b'):
            frame_index = max(frame_index - 10, 0)

        if paused:
            while True:
                key2 = cv2.waitKey(0) & 0xFF
                if key2 == ord(' '):
                    paused = not paused
                    break
                elif key2 == ord('f'):
                    frame_index = min(frame_index + 10, total_frames - 1)
                elif key2 == ord('b'):
                    frame_index = max(frame_index - 10, 0)
                elif key2 == 27:
                    cap_original.release()
                    cap_pose.release()
                    cv2.destroyAllWindows()
                    return
                
                cap_original.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                cap_pose.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ret1, frame_original = cap_original.read()
                ret2, frame_pose = cap_pose.read()

                if not ret1 or not ret2:
                    break

                h, w, _ = frame_pose.shape
                landmarks = data[frame_index][1:]
                for i in range(0, len(landmarks), 2):
                    x = int(landmarks[i] * w)
                    y = int(landmarks[i + 1] * h)
                    cv2.circle(frame_pose, (x, y), 3, (0, 255, 0), -1)
                    cv2.putText(frame_pose, f"{i//2}: ({x}, {y})", (x, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                
                result = np.hstack((frame_original, frame_pose))
                cv2.imshow('moto | Pose PlayBack', result)
        else:
            frame_index += 1

    replay_again = input("再生が終了しました。もう一度再生しますか？ (y/n): ")
    if replay_again.lower() == 'y':
        playback_pose()

def video_mode():
    name=input("選択するビデオのファイル名を入力：")

    if not name.lower().endswith(".mp4"):
        print("ファイル形式が間違えています．")
    else:
        cap_original = cv2.VideoCapture(name)

        if not cap_original.isOpened():
            print("ファイル名を確認してください．")
            return
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = cap_original.get(cv2.CAP_PROP_FPS)
        width = int(cap_original.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap_original.get(cv2.CAP_PROP_FRAME_HEIGHT))

        cap_pose = cv2.VideoWriter('recorded_pose.mp4', fourcc, fps, (width, height))
        csv_file = open('landmarks.csv', 'w', newline='')
        csv_writer = csv.writer(csv_file)

        pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        while cap_original.isOpened():
            ret, frame_original = cap_original.read()
            if not ret:
                break

            frame_pose = frame_original.copy()
            image = cv2.cvtColor(frame_pose, cv2.COLOR_BGR2RGB)
            results = pose.process(image)

            landmarks_frame = []
            if results.pose_landmarks:
                landmarks_frame.append(time.time())
                for landmark in results.pose_landmarks.landmark:
                    landmarks_frame.extend([landmark.x, landmark.y])

            csv_writer.writerow(landmarks_frame)

            mp_drawing.draw_landmarks(frame_pose, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            h, w, _ = frame_pose.shape
            if results.pose_landmarks:
                for idx, landmark in enumerate(results.pose_landmarks.landmark):
                    x = int(landmark.x * w)
                    y = int(landmark.y * h)
                    cv2.circle(frame_pose, (x, y), 3, (0, 255, 0), -1)  # 点を描画
                    cv2.putText(frame_pose, f"{idx}:({x},{y})", (x, y), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

            cap_pose.write(frame_pose)

            result = np.hstack((frame_original, frame_pose))
            cv2.imshow('Original | Pose Detection', result)

            if cv2.waitKey(1) & 0xFF == 27:
                break

        cap_original.release()
        cap_pose.release()
        csv_file.close()
        cv2.destroyAllWindows()

        playback_pose()

        

if __name__ == '__main__':
    mode = input("モード選択: 1=カメラモード, 2=ビデオモード > ")

    if mode == '1':
        record_pose()
    elif mode == '2':
        video_mode()
    else:
        print("無効な入力です。");