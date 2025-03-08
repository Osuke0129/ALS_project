import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose
# model_complexityを2にすることで、より精度高く全身検出可能（処理負荷は増える）
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=2,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)  # カメラを開く

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False

    # ポーズ検出を行う
    results = pose.process(image)

    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # 骨格を描画（全身）
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(0,0,255), thickness=2)
        )

    cv2.imshow("MediaPipe Pose - Full Body", image)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
