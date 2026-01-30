import cv2
import mediapipe as mp
import numpy as np
import pyk4a
from pyk4a import Config, PyK4A, ColorResolution, DepthMode, FPS, CalibrationType
import open3d as o3d
import sys

# --- ฟังก์ชันช่วยดึงค่า 3D (แยกออกมาให้โค้ดสะอาด) ---
def get_safe_3d(lm, depth_map, calibration, w, h):
    """ ดึงค่าพิกัด 3D แบบปลอดภัย ไม่ให้โปรแกรม Error """
    try:
        cx, cy = int(lm.x * w), int(lm.y * h)
        # เช็คขอบเขตภาพ
        if cx < 0 or cx >= w or cy < 0 or cy >= h:
            return None, (cx, cy)
        
        # ดึงค่าความลึก
        d = depth_map[cy, cx]
        if d <= 0: 
            return None, (cx, cy)

        # แปลงเป็น 3D
        pt3d = calibration.convert_2d_to_3d((cx, cy), float(d), CalibrationType.COLOR)
        return pt3d, (cx, cy)
    except:
        return None, (0,0)

def main():
    # 1. System Setup
    try:
        config = Config(
            color_resolution=ColorResolution.RES_720P,
            depth_mode=DepthMode.NFOV_UNBINNED,
            camera_fps=FPS.FPS_30,
            synchronized_images_only=True
        )
        k4a = PyK4A(config)
        k4a.start()
        print(">>> Camera Initialized. Starting...")
    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
        return

    # Setup MediaPipe
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    mp_drawing = mp.solutions.drawing_utils

    # ตัวแปรระบบ
    plane_eqn = None
    calib_points = []
    is_calibrated = False

    print(">>> System Running. Please wait for floor calibration...")

    while True:
        try:
            # ดึงภาพ
            capture = k4a.get_capture()
            if capture.color is None or capture.transformed_depth is None:
                continue

            # --- แก้ไข Memory Layout (กัน OpenCV Error) ---
            color_img = np.ascontiguousarray(capture.color[:, :, :3])
            depth_map = capture.transformed_depth
            
            # --- PHASE A: หาพื้น (Auto Calibration) ---
            if not is_calibrated:
                cv2.putText(color_img, "INITIALIZING...", (50, 300), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                
                # เก็บจุดตัวอย่างหาพื้น
                try:
                    points = capture.depth_point_cloud.reshape(-1, 3)
                    # กรองระยะ 1-3 เมตร
                    mask = (points[:, 2] > 1000) & (points[:, 2] < 3000)
                    valid_points = points[mask]
                    
                    if len(valid_points) > 0:
                        calib_points.extend(valid_points[::100])
                    
                    # คำนวณ RANSAC
                    if len(calib_points) > 200:
                        pcd = o3d.geometry.PointCloud()
                        pcd.points = o3d.utility.Vector3dVector(np.array(calib_points))
                        plane, _ = pcd.segment_plane(distance_threshold=15, ransac_n=3, num_iterations=200)
                        
                        # เช็คว่าเป็นพื้นแนวนอนหรือไม่
                        if abs(plane[1]) > 0.8:
                            plane_eqn = plane
                            is_calibrated = True
                            print(f">>> Floor Locked: {plane}")
                        else:
                            calib_points = [] # รีเซ็ตหาใหม่
                except Exception:
                    pass # ข้ามไปถ้ามีปัญหาในการคำนวณ Point Cloud

            # --- PHASE B: ตรวจจับและวัดค่า (เมื่อเจอพื้นแล้ว) ---
            else:
                img_rgb = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
                results = pose.process(img_rgb)
                h, w, _ = color_img.shape

                if results.pose_landmarks:
                    # เรียกฟังก์ชันที่แยกออกไปด้านนอก Loop (ลดปัญหา Syntax Error)
                    pt_L, pos_L = get_safe_3d(results.pose_landmarks.landmark[31], depth_map, k4a.calibration, w, h)
                    pt_R, pos_R = get_safe_3d(results.pose_landmarks.landmark[32], depth_map, k4a.calibration, w, h)

                    if pt_L is not None and pt_R is not None:
                        A, B, C, D = plane_eqn
                        denom = np.sqrt(A**2 + B**2 + C**2)
                        
                        # 1. คำนวณระยะดิบจากระนาบ
                        raw_h_L = abs(A*pt_L[0] + B*pt_L[1] + C*pt_L[2] + D) / denom
                        raw_h_R = abs(A*pt_R[0] + B*pt_R[1] + C*pt_R[2] + D) / denom

                        # 2. *** AUTO-ZERO LOGIC ***
                        # ใช้เท้าข้างที่ต่ำสุดเป็น Bias เพื่อแก้ Z-Drift
                        floor_bias = min(raw_h_L, raw_h_R)

                        final_h_L = raw_h_L - floor_bias
                        final_h_R = raw_h_R - floor_bias

                        # ตัด Noise
                        if final_h_L < 8.0: final_h_L = 0.0
                        if final_h_R < 8.0: final_h_R = 0.0

                        # แสดงผล
                        col_L = (0, 255, 0) if final_h_L > 15 else (200, 200, 200)
                        col_R = (0, 255, 0) if final_h_R > 15 else (200, 200, 200)

                        cv2.putText(color_img, f"L:{int(final_h_L)}", pos_L, cv2.FONT_HERSHEY_SIMPLEX, 0.6, col_L, 2)
                        cv2.putText(color_img, f"R:{int(final_h_R)}", pos_R, cv2.FONT_HERSHEY_SIMPLEX, 0.6, col_R, 2)

                        # Dashboard
                        cv2.rectangle(color_img, (10, 10), (350, 140), (40, 40, 40), -1)
                        cv2.putText(color_img, "MFC MONITOR", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                        cv2.putText(color_img, f"L Height: {final_h_L:.1f} mm", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col_L, 2)
                        cv2.putText(color_img, f"R Height: {final_h_R:.1f} mm", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col_R, 2)

                    mp_drawing.draw_landmarks(color_img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            cv2.imshow("Kinect MFC System", color_img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        except Exception as e:
            # พิมพ์ Error เล็กน้อยแต่ไม่หยุดโปรแกรม
            # print(f"Frame Skipped: {e}") 
            continue

    k4a.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()