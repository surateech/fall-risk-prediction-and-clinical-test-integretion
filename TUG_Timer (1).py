import cv2
import mediapipe as mp
import pyk4a
from pyk4a import Config, PyK4A
import time
import numpy as np

# =================================================================
# 🔧 ส่วนตั้งค่า (CONFIGURATION)
# =================================================================
CHAIR_X = 1133   
CHAIR_Y = 584   
STAND_OFFSET = 50 
# =================================================================

# --- State Definitions ---
STATE_WAITING = 0   
STATE_ACTIVE = 1    
STATE_FINISHED = 2  

# --- Global Variables ---
current_state = STATE_WAITING
start_time = 0
end_time = 0
should_exit = False

def mouse_callback(event, x, y, flags, param):
    global current_state, start_time, end_time, should_exit
    if event == cv2.EVENT_LBUTTONDOWN:
        if x > 1150 and y < 80:
            should_exit = True
            print(">>> STOP BUTTON CLICKED <<<")
    elif event == cv2.EVENT_RBUTTONDOWN:
        current_state = STATE_WAITING
        start_time = 0
        end_time = 0
        print(">>> RESET TEST <<<")

def main():
    global current_state, start_time, end_time, should_exit
    
    # 1. Setup Azure Kinect (แก้ไขโหมด Depth ให้กว้างขึ้น)
    config = Config(
        color_resolution=pyk4a.ColorResolution.RES_720P,
        # --- แก้ไขจุดที่ 1: เปลี่ยนเป็น WFOV (มุมกว้าง) เพื่อให้ภาพเต็มจอ ---
        depth_mode=pyk4a.DepthMode.WFOV_2X2BINNED, 
        # -----------------------------------------------------------
        camera_fps=pyk4a.FPS.FPS_30,
        synchronized_images_only=True,
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

    window_name = 'Azure Kinect TUG - Depth Fullscreen'
    
    # --- แก้ไขจุดที่ 2: ตั้งค่าหน้าต่างให้เต็มจอ (Fullscreen) ---
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    # -----------------------------------------------------
    
    cv2.setMouseCallback(window_name, mouse_callback)

    try:
        while not should_exit:
            capture = k4a.get_capture()
            if capture.color is None or capture.transformed_depth is None:
                continue

            # --- Image Processing ---
            frame_rgb = cv2.cvtColor(capture.color[:, :, :3], cv2.COLOR_BGR2RGB)
            
            # เตรียมภาพ Depth
            depth_map = capture.transformed_depth 
            
            # Normalize ให้เห็นชัดขึ้น (ตัดค่าที่ไกลเกิน 3 เมตรออกเพื่อให้ contrast ดีขึ้น)
            # ถ้าไม่ตัด ค่าความลึกของห้องที่กว้างมากจะทำให้ตัวคนมืด
            depth_clipped = np.clip(depth_map, 0, 3000) # clip ที่ 3 เมตร (3000mm)
            depth_display = cv2.normalize(depth_clipped, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            frame_display = cv2.applyColorMap(depth_display, cv2.COLORMAP_JET)

            h, w, c = frame_display.shape

            # Process AI
            results = pose.process(frame_rgb)

            # === วาดกราฟิก ===
            cv2.rectangle(frame_display, (w - 130, 10), (w - 10, 60), (0, 0, 200), -1)
            cv2.putText(frame_display, "STOP", (w - 110, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            threshold_line_y = CHAIR_Y - STAND_OFFSET

            cv2.circle(frame_display, (CHAIR_X, CHAIR_Y), 10, (0, 255, 0), -1)
            cv2.putText(frame_display, f"Chair", (CHAIR_X + 15, CHAIR_Y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            cv2.line(frame_display, (0, threshold_line_y), (w, threshold_line_y), (0, 255, 255), 2)
            cv2.putText(frame_display, "Stand Line", (20, threshold_line_y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

            message = "Ready - Waiting for Stand up"

            if results.pose_landmarks:
                mp_drawing.draw_landmarks(frame_display, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                left_hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
                right_hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]
                
                hip_x = int((left_hip.x + right_hip.x) * w / 2)
                hip_y = int((left_hip.y + right_hip.y) * h / 2)

                # --- Logic ---
                if current_state == STATE_WAITING:
                    message = "Ready. Stand up to start."
                    if hip_y < threshold_line_y:
                        start_time = time.time()
                        current_state = STATE_ACTIVE
                        print("Timer Started!")

                elif current_state == STATE_ACTIVE:
                    elapsed = time.time() - start_time
                    message = f"GO! Time: {elapsed:.2f} s"
                    
                    if elapsed > 3.0 and hip_y > threshold_line_y:
                        end_time = time.time()
                        current_state = STATE_FINISHED
                        print("Timer Stopped!")

                elif current_state == STATE_FINISHED:
                    total_time = end_time - start_time
                    message = f"DONE! Total: {total_time:.2f} s"
                    cv2.putText(frame_display, "Right Click to Reset", (20, 150),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

                cv2.circle(frame_display, (hip_x, hip_y), 8, (0, 0, 255), -1)

            # UI Header
            cv2.rectangle(frame_display, (0, 0), (w - 140, 80), (0, 0, 0), -1)
            cv2.putText(frame_display, message, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)

            cv2.imshow(window_name, frame_display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    finally:
        k4a.stop()
        cv2.destroyAllWindows()
        print("Program closed.")

if __name__ == "__main__":
    main()