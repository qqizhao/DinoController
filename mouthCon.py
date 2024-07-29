import cv2
import mediapipe as mp
import numpy as np
from pynput.keyboard import Key, Controller

def mouth_aspect_ratio(mouth):
    # 计算嘴部长宽比
    A = np.linalg.norm(np.array(mouth[2]) - np.array(mouth[3]))  # 81, 178
    B = np.linalg.norm(np.array(mouth[4]) - np.array(mouth[5]))  # 311, 402
    C = np.linalg.norm(np.array(mouth[0]) - np.array(mouth[1]))  # 61, 291
    mar = (A + B) / (2.0 * C)
    return mar

def detect_mouth_open():
    mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        min_detection_confidence=0.5)

    # 嘴部关键点索引
    mouth_indices = [61,291,81,178,311,402]

    mouth_ar_thresh = 0.3
    
    keyboard = Controller()
    
    # 视频流初始化
    cap = cv2.VideoCapture(0)

    while True:
        success, image = cap.read()
        if not success:
            break

        # 处理图像
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = mp_face_mesh.process(image_rgb)

        # 提取嘴部关键点
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0].landmark
            mouth = []
            for idx in mouth_indices:
                x = int(face_landmarks[idx].x * image.shape[1])
                y = int(face_landmarks[idx].y * image.shape[0])
                mouth.append((x, y))

            # 计算嘴部长宽比
            mar = mouth_aspect_ratio(mouth)

            # 绘制嘴部区域和嘴部长宽比
            mouth_hull = np.array(mouth, dtype=np.int32)
            cv2.polylines(image, [mouth_hull], isClosed=True, color=(0, 255, 0), thickness=1)
            cv2.putText(image, f"MAR: {mar:.2f}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # 判断嘴部是否张开
            if mar > mouth_ar_thresh:
                keyboard.press(Key.space)
                keyboard.release(Key.space)
                cv2.putText(image, "Mouth is Open!", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


        cv2.imshow("MediaPipe Face Mesh", image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # realease
    cap.release()
    cv2.destroyAllWindows()
    mp_face_mesh.close()

if __name__ == "__main__":

    detect_mouth_open()
