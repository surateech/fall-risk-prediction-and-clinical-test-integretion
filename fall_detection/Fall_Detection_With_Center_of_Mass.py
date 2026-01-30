import cv2
import mediapipe as mp
import numpy as np
import pyk4a
from pyk4a import Config, PyK4A, CalibrationType

# --- Global Variables for Control ---
should_exit = False
window_name_main = "Kinect Fullscreen Depth"

def mouse_callback(event, x, y, flags, param):
    """ฟังก์ชันจัดการเมาส์: คลิกปุ่ม STOP เพื่อปิด"""
    global should_exit
    if event == cv2.EVENT_LBUTTONDOWN:
        # เช็คตำแหน่งคลิก (มุมขวาบน)
        # สมมติความกว้างจอ 720p (1280px) ปุ่มอยู่ขวาสุดประมาณ 150px
        if x > 1100 and y < 80:
            should_exit = True
            print(">>> STOP BUTTON CLICKED <<<")

# --- 1. MediaPipe Setup ---
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# --- 2. Body Segment Parameters ---
SEGMENTS = [
    {'name': 'Head', 'mass': 0.081, 'joints': [mp_pose.PoseLandmark.NOSE, mp_pose.PoseLandmark.LEFT_EAR]},
    {'name': 'Torso', 'mass': 0.497, 'joints': [mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_HIP]},
    {'name': 'R_UpperArm', 'mass': 0.028, 'joints': [mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW]},
    {'name': 'L_UpperArm', 'mass': 0.028, 'joints': [mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW]},
    {'name': 'R_Forearm', 'mass': 0.016, 'joints': [mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_WRIST]},
    {'name': 'L_Forearm', 'mass': 0.016, 'joints': [mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_WRIST]},
    {'name': 'R_Hand', 'mass': 0.006, 'joints': [mp_pose.PoseLandmark.RIGHT_WRIST, mp_pose.PoseLandmark.RIGHT_INDEX]},
    {'name': 'L_Hand', 'mass': 0.006, 'joints': [mp_pose.PoseLandmark.LEFT_WRIST, mp_pose.PoseLandmark.LEFT_INDEX]},
    {'name': 'R_Thigh', 'mass': 0.100, 'joints': [mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_KNEE]},
    {'name': 'L_Thigh', 'mass': 0.100, 'joints': [mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_KNEE]},
    {'name': 'R_Shin', 'mass': 0.0465, 'joints': [mp_pose.PoseLandmark.RIGHT_KNEE, mp_pose.PoseLandmark.RIGHT_HEEL]},
    {'name': 'L_Shin', 'mass': 0.0465, 'joints': [mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.LEFT_HEEL]},
    {'name': 'R_Foot', 'mass': 0.0145, 'joints': [mp_pose.PoseLandmark.RIGHT_HEEL, mp_pose.PoseLandmark.RIGHT_FOOT_INDEX]},
    {'name': 'L_Foot', 'mass': 0.0145, 'joints': [mp_pose.PoseLandmark.LEFT_HEEL, mp_pose.PoseLandmark.LEFT_FOOT_INDEX]},
]

# --- 3. Azure Kinect Setup ---
k4a = PyK4A(
    Config(
        color_resolution=pyk4a.ColorResolution.RES_720P,
        depth_mode=pyk4a.DepthMode.WFOV_2X2BINNED, 
        camera_fps=pyk4a.FPS.FPS_30,
        synchronized_images_only=True,
    )
)
k4a.start()
calibration = k4a.calibration

def get_joint_3d_real(landmarks, joint_idx, depth_map, width, height):
    lm = landmarks[joint_idx]
    x_pixel = int(lm.x * width)
    y_pixel = int(lm.y * height)

    # 1. เช็คขอบเขตภาพ
    if x_pixel < 0 or x_pixel >= width or y_pixel < 0 or y_pixel >= height:
        return None

    # 2. เช็คค่า Depth เป็น 0 หรือไม่
    depth_mm = depth_map[y_pixel, x_pixel]
    if depth_mm == 0:
        return None

    # 3. [แก้ไข] ใส่ Try-Except ป้องกัน Error เวลาพิกัดอยู่ขอบเลนส์มากๆ
    try:
        point3d_mm = calibration.convert_2d_to_3d((x_pixel, y_pixel), depth_mm, CalibrationType.COLOR)
        return np.array(point3d_mm) / 1000.0
    except ValueError:
        # กรณีที่ Kinect Calibration model คำนวณค่าตรงจุดนั้นไม่ได้
        return None

def calculate_true_com(landmarks, depth_map, width, height):
    total_mass = 0
    com_vec = np.zeros(3)
    for seg in SEGMENTS:
        p1 = get_joint_3d_real(landmarks, seg['joints'][0].value, depth_map, width, height)
        p2 = get_joint_3d_real(landmarks, seg['joints'][1].value, depth_map, width, height)
        if p1 is None or p2 is None:
            continue
        segment_center = (p1 + p2) / 2.0
        com_vec += segment_center * seg['mass']
        total_mass += seg['mass']
    if total_mass > 0:
        return com_vec / total_mass
    return None

def get_bos_circle_real(landmarks, depth_map, width, height):
    l_heel = get_joint_3d_real(landmarks, mp_pose.PoseLandmark.LEFT_HEEL.value, depth_map, width, height)
    r_heel = get_joint_3d_real(landmarks, mp_pose.PoseLandmark.RIGHT_HEEL.value, depth_map, width, height)
    if l_heel is None or r_heel is None:
        return None, None
        
    center_x = (l_heel[0] + r_heel[0]) / 2.0
    center_z = (l_heel[2] + r_heel[2]) / 2.0
    
    dist = np.linalg.norm(l_heel - r_heel)
    radius = (dist / 2.0) + 0.08
    
    return (center_x, center_z), radius

def draw_top_down(com_real, bos_center, bos_radius):
    W, H = 400, 400 
    canvas = np.zeros((H, W, 3), dtype=np.uint8)
    grid_color = (50, 50, 50)
    
    scale = 50 
    cx = W // 2
    cy = H - 30 
    
    max_dist_m = int(H / scale) + 1
    for i in range(1, max_dist_m):
        radius_px = int(i * scale)
        cv2.ellipse(canvas, (cx, cy), (radius_px, radius_px), 0, 180, 360, grid_color, 1)
        cv2.putText(canvas, f"{i}m", (cx + 5, cy - radius_px + 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
    
    cv2.circle(canvas, (cx, cy), 10, (100, 100, 100), -1)
    cv2.putText(canvas, "Cam", (cx-15, cy+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150,150,150), 1)

    if com_real is None or bos_center is None:
        return canvas

    bx_px = int(bos_center[0] * scale) + cx
    bz_px = cy - int(bos_center[1] * scale)
    br_px = int(bos_radius * scale)
    
    cv2.circle(canvas, (bx_px, bz_px), br_px, (0, 255, 0), 2)
    
    cmx_px = int(com_real[0] * scale) + cx
    cmz_px = cy - int(com_real[2] * scale)
    
    dist_xz = np.sqrt((com_real[0] - bos_center[0])**2 + (com_real[2] - bos_center[1])**2)
    is_stable = dist_xz <= bos_radius
    
    color = (0, 255, 255) if is_stable else (0, 0, 255)
    
    cv2.line(canvas, (bx_px, bz_px), (cmx_px, cmz_px), (100, 100, 100), 1)
    cv2.circle(canvas, (cmx_px, cmz_px), 8, color, -1)
    
    cv2.putText(canvas, f"CoM Height: {com_real[1]:.2f}m", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    
    margin = bos_radius - dist_xz
    cv2.putText(canvas, f"Margin: {margin:.2f}m", (15, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    
    status_text = "BALANCED" if is_stable else "FALL DETECTED"
    cv2.putText(canvas, status_text, (15, H - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return canvas

# --- Main Setup ---
cv2.namedWindow(window_name_main, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(window_name_main, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
cv2.setMouseCallback(window_name_main, mouse_callback)

try:
    print("Starting Fall Detection (Depth Fullscreen)...")
    while not should_exit:
        capture = k4a.get_capture()
        if capture.color is not None and capture.transformed_depth is not None:
            
            frame_rgb = cv2.cvtColor(capture.color[:, :, :3], cv2.COLOR_BGR2RGB)
            depth_map = capture.transformed_depth 
            h, w = depth_map.shape

            depth_clipped = np.clip(depth_map, 0, 3000)
            depth_display = cv2.normalize(depth_clipped, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            depth_colormap = cv2.applyColorMap(depth_display, cv2.COLORMAP_JET)

            # Draw STOP Button
            cv2.rectangle(depth_colormap, (w - 130, 10), (w - 10, 60), (0, 0, 200), -1)
            cv2.putText(depth_colormap, "STOP", (w - 110, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            results = pose.process(frame_rgb)
            top_view = np.zeros((400, 400, 3), dtype=np.uint8)

            if results.pose_landmarks:
                real_com = calculate_true_com(results.pose_landmarks.landmark, depth_map, w, h)
                bos_center, bos_radius = get_bos_circle_real(results.pose_landmarks.landmark, depth_map, w, h)
                
                mp_drawing.draw_landmarks(depth_colormap, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                
                if real_com is not None and bos_center is not None:
                    top_view = draw_top_down(real_com, bos_center, bos_radius)
                    cv2.putText(depth_colormap, f"Z: {real_com[2]:.2f}m", (20, 50), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            cv2.imshow(window_name_main, depth_colormap)
            cv2.imshow("Analysis (Top-Down)", top_view)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    pass
finally:
    k4a.stop()
    cv2.destroyAllWindows()
    print("Program closed.")