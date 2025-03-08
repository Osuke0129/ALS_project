import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=2,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False

    results = pose.process(image)

    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.pose_landmarks:
        # 骨格を描画
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(0,0,255), thickness=2)
        )

        # 全ランドマークの座標を表示する
        for idx, landmark in enumerate(results.pose_landmarks.landmark):
            h, w, _ = image.shape
            x, y = int(landmark.x * w), int(landmark.y * h)
            coord_text = f"{idx}: ({x}, {y})"
            cv2.putText(image, coord_text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.imshow('Pose Estimation', image)

    if cv2.waitKey(5) & 0xFF == 27:  # ESCキーで終了
        break

cap.release()
cv2.destroyAllWindows()
