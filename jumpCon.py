import cv2
import mediapipe as mp
from pynput.keyboard import Key, Controller

def nose_detection():

    cap = cv2.VideoCapture(0)  # 打开摄像头

    threshold = 7  # 阈值，可以根据实际情况调整
    
    prev_nose_y = None
    keyboard = Controller()
    
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read() 
            if not ret:
                continue

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)

            # 绘制图像
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                # 获取鼻子的关键点
                nose = results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]
                nose_x, nose_y = int(nose.x * frame.shape[1]), int(nose.y * frame.shape[0])

            
                # 计算鼻子高度变化
                if prev_nose_y is not None:
                    nose_height_change = nose_y - prev_nose_y
                    if nose_height_change > threshold:
                        keyboard.press(Key.space)
                        keyboard.release(Key.space)
                        cv2.putText(image, "Jump", (110,620), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255),2)  # 在帧上标注"口部打开"信息
                        print("Jump")
                        
                prev_nose_y = nose_y

            cv2.imshow('MediaPipe Pose', image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    nose_detection()