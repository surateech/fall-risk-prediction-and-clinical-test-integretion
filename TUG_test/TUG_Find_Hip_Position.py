import cv2
import mediapipe as mp
import numpy as np
import pyk4a
from pyk4a import Config, PyK4A
import time

# --- Configuration ---
STAND_OFFSET_PIXEL = 50    # ต้องลุกสูงกว่าจุดที่คลิกกี่พิกเซลถึงจะเริ่มนับ

# --- State Definitions ---
STATE_SETUP = -1    # รอคลิกตั้งค่า
STATE_WAITING = 0   # นั่งรอเริ่ม (Ready)
STATE_ACTIVE = 1    # กำลังทดสอบ (Standing/Walking)
STATE_FINISHED = 2  # จบการทดสอบ (Finished)

# --- Global Variables ---
current_state = STATE_SETUP
start_time = 0
end_time = 0

chair_point = None      # เก็บพิกัด (x, y) ที่คลิก
is_chair_set = False
should_exit = False     # ตัวแปรควบคุมการปิดโปรแกรม

def mouse_callback(event, x, y, flags, param):
    """ฟังก์ชันจัดการเมาส์"""
    global chair_point, is_chair_set, current_state, start_time, end_time, should_exit

    if event == cv2.EVENT_LBUTTONDOWN:
        # 1. เช็คปุ่ม STOP (มุมขวาบน)
        # ตรวจสอบพิกัดคร่าวๆ (สมมติปุ่มอยู่โซนขวาบน x > 1150)
        if x > 1150 and y < 80:
            should_exit = True
            print(">>> STOP BUTTON CLICKED <<<")
            return

        # 2. ตั้งค่าจุดเก้าอี้ (Chair Point)
        chair_point = (x, y)
        is_chair_set = True
        current_state = STATE_WAITING
        print(f"Set Chair Point at: {chair_point}")

    elif event == cv2.EVENT_RBUTTONDOWN:
        # คลิกขวา Reset
        if is_chair_set:
            current_state = STATE_WAITING
            start_time = 0
            end_time = 0
            print(">>> RESET TEST <<<")

def main():
    global current_state, start_time, end_time, chair_point, is_chair_set, should_exit
    
    # 1. Setup Azure Kinect
    config = Config(
        color_resolution=pyk4a.ColorResolution.RES_720P,
        depth_mode=pyk4a.DepthMode.OFF, # ปิด Depth mode ไปเลยเพราะไม่ใช้แล้ว
        camera_fps=pyk4a.FPS.FPS_30,
        synchronized_images_only=False, # ไม่ต้อง sync เพราะใช้แค่ภาพสี
    )
    k4a = PyK4A(config)
    k4a.start()

    # 2. Setup MediaPipe Pose
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    mp_drawing = mp.solutions.drawing_utils

    window_name = 'Azure Kinect TUG - Position Only'
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)

    try:
        while not should_exit:
            capture = k4a.get_capture()
            if capture.color is None:
                continue

            # Image Processing
            frame = capture.color[:, :, :3] # เอาเฉพาะ RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # MediaPipe Process
            results = pose.process(frame_rgb)
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            h, w, c = frame_bgr.shape

            # === วาดปุ่ม STOP ===
            cv2.rectangle(frame_bgr, (w - 130, 10), (w - 10, 60), (0, 0, 200), -1)
            cv2.putText(frame_bgr, "STOP", (w - 110, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            # === วาด Setup UI & Coordinates ===
            stand_threshold_y = 0
            
            if is_chair_set and chair_point:
                cx, cy = chair_point
                stand_threshold_y = cy - STAND_OFFSET_PIXEL
                
                # 1. วาดจุดเขียว
                cv2.circle(frame_bgr, (cx, cy), 8, (0, 255, 0), -1)
                # 2. วาดเส้น Threshold
                cv2.line(frame_bgr, (0, stand_threshold_y), (w, stand_threshold_y), (0, 255, 255), 2)
                
                # 3. [NEW] แสดงพิกัด (X, Y) บนหน้าจอ
                coord_text = f"Pos: ({cx}, {cy})"
                cv2.putText(frame_bgr, coord_text, (cx + 15, cy + 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            message = "Left Click Hips to Setup"

            if results.pose_landmarks:
                mp_drawing.draw_landmarks(frame_bgr, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                # หาตำแหน่งสะโพก (Hip Center)
                left_hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
                right_hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]
                
                hip_x = int((left_hip.x + right_hip.x) * w / 2)
                hip_y = int((left_hip.y + right_hip.y) * h / 2)

                # --- State Machine Logic (Time Only) ---
                if current_state == STATE_WAITING:
                    message = "Ready. Stand up to start."
                    if hip_y < stand_threshold_y: # ลุกยืน (Y น้อยลง = สูงขึ้น)
                        start_time = time.time()
                        current_state = STATE_ACTIVE

                elif current_state == STATE_ACTIVE:
                    elapsed = time.time() - start_time
                    message = f"GO! Time: {elapsed:.2f} s"
                    
                    # Logic จบ: ต้องเดินไปแล้วกลับมานั่ง (Hip ต่ำลงมาเกินเส้น)
                    # เพิ่มเงื่อนไข time > 3 วินาที เพื่อกันกรณียืนๆ นั่งๆ ทันที (False Finish)
                    if hip_y > stand_threshold_y and elapsed > 3.0:
                        end_time = time.time()
                        current_state = STATE_FINISHED

                elif current_state == STATE_FINISHED:
                    total_time = end_time - start_time
                    message = f"DONE! Total: {total_time:.2f} s"
                    cv2.putText(frame_bgr, "Right Click to Reset", (20, 150),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

                # แสดงจุดสะโพกปัจจุบัน
                cv2.circle(frame_bgr, (hip_x, hip_y), 5, (0, 0, 255), -1)

            # UI Header
            cv2.rectangle(frame_bgr, (0, 0), (w - 140, 80), (0, 0, 0), -1)
            cv2.putText(frame_bgr, message, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)

            cv2.imshow(window_name, frame_bgr)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    finally:
        k4a.stop()
        cv2.destroyAllWindows()
        print("Program closed.")

if __name__ == "__main__":
    main()