import os
import cv2
from datetime import datetime
from ultralytics import YOLO


username = input("名前を入力してください: ")
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
base_dir = r"C:\Users\217ki\ALS_project\eye_info"
output_dir = os.path.join(base_dir, f"{username}_{timestamp}")
os.makedirs(output_dir, exist_ok=True)
print(f"[INFO] 解析結果は以下に保存されます: {output_dir}")


video_name = input("解析したい.mp4ファイル名（例: sample.mp4）を入力してください: ")
video_dir = r"C:\Users\217ki\ALS_project\eye_data"
video_path = os.path.join(video_dir, video_name)

if not os.path.exists(video_path):
    print("[ERROR] 指定された動画ファイルが存在しません。終了します。")
    exit()


model = YOLO("yolov8n.pt") 

cap = cv2.VideoCapture(video_path)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out_path = os.path.join(output_dir, "result.mp4")
out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

print("[INFO] YOLO解析を開始します。途中で終了するには 'q' を押してください。")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break


    results = model.predict(source=frame, conf=0.5, verbose=False)
    annotated_frame = results[0].plot()

    center_x, center_y = width // 2, height // 2
    cv2.circle(annotated_frame, (center_x, center_y), radius=8, color=(0, 0, 255), thickness=-1)

    out.write(annotated_frame)
    cv2.imshow("YOLO解析中", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("[INFO] 解析を中断しました。")
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print(f"[DONE] 解析完了！保存先: {out_path}")
