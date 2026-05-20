import cv2
import numpy as np
import mediapipe as mp
import pykinect_azure as pykinect
import time
import math
import csv
from datetime import datetime
from collections import deque

# ==========================================
# Global Variables for Floor Plane Setup
# ==========================================
floor_clicks = []
floor_points_3d = []
floor_normal = None
floor_centroid = None

def mouse_callback(event, x, y, flags, param):
    global floor_clicks
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(floor_clicks) < 4:
            floor_clicks.append((x, y))

def get_3d_point(x_norm, y_norm, depth_image):
    """
    Converts normalized 2D coordinates (from mediapipe) to 3D space using the depth map.
    Returns [X, Y, Z] in millimeters.
    """
    h, w = depth_image.shape
    px = int(x_norm * w)
    py = int(y_norm * h)
    
    # Boundary constraints
    px = max(0, min(px, w - 1))
    py = max(0, min(py, h - 1))
    
    # Retrieve depth value (Z)
    z = depth_image[py, px]
    
    # Handle zero depth (holes in depth map) by taking a median from a 21x21 window
    if z == 0:
        window = depth_image[max(0, py-10):min(h, py+11), max(0, px-10):min(w, px+11)]
        non_zeros = window[window > 0]
        if len(non_zeros) > 0:
            z = np.median(non_zeros)
            
    if z == 0:
        return None
        
    # Approximate intrinsic parameters for 720p Azure Kinect Color Camera
    cx, cy = w / 2.0, h / 2.0
    fx, fy = 600.0, 600.0 
    
    x = (px - cx) * z / fx
    y = (py - cy) * z / fy
    
    return np.array([x, y, z])

def fit_plane(points_3d):
    """
    Calculates the plane normal and centroid from a set of 3D points using SVD.
    """
    centroid = np.mean(points_3d, axis=0)
    centered = points_3d - centroid
    _, _, vh = np.linalg.svd(centered)
    normal = vh[2, :]
    return normal, centroid

def draw_risk_graph(image, risk_history, x, y, w, h):
    """
    Draws a real-time line graph of the fall risk on the screen.
    """
    # Draw semi-transparent background
    overlay = image.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.6, image, 0.4, 0, image)

    # Draw border and labels
    cv2.rectangle(image, (x, y), (x + w, y + h), (200, 200, 200), 1)
    cv2.putText(image, "Fall Risk (%) vs Time", (x + 5, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.putText(image, "100", (x - 25, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.putText(image, "0", (x - 15, y + h), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    # Draw horizontal middle line (50% risk marker)
    cv2.line(image, (x, y + int(h/2)), (x + w, y + int(h/2)), (100, 100, 100), 1)

    # Plot the graph line
    if len(risk_history) > 1:
        pts = []
        x_step = w / (risk_history.maxlen - 1)
        
        start_x = x + w - (len(risk_history) - 1) * x_step
        
        for i, val in enumerate(risk_history):
            px = int(start_x + i * x_step)
            py = int(y + h - (val / 100.0) * h)
            pts.append([px, py])
        
        pts = np.array(pts, np.int32).reshape((-1, 1, 2))
        
        latest_risk = risk_history[-1]
        line_color = (0, 0, 255) if latest_risk > 50.0 else (0, 255, 0)
        
        cv2.polylines(image, [pts], False, line_color, 2)

def main():
    global floor_clicks, floor_points_3d, floor_normal, floor_centroid

    # Initialize Azure Kinect
    pykinect.initialize_libraries()
    device_config = pykinect.default_configuration
    device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_720P
    device_config.depth_mode = pykinect.K4A_DEPTH_MODE_NFOV_UNBINNED
    device_config.camera_fps = pykinect.K4A_FRAMES_PER_SECOND_30
    device = pykinect.start_device(config=device_config)

    # Initialize MediaPipe Pose
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils

    # Variables for Calculations & History
    prev_com_y = None
    prev_time = time.time()
    fall_risk_history = deque(maxlen=100)

    # Variables for CSV Logging
    is_logging = False
    log_start_time = 0.0
    log_file = None
    csv_writer = None

    # Fall Risk Constants
    B_0 = 0.0
    B_1 = 5.0
    B_2 = 0.04
    B_3 = 1.0

    #EMA
    ema_alpha = 0.3
    downward_velocity = 0.0

    # Setup OpenCV Window and Mouse Callback
    cv2.namedWindow("Azure Kinect Biomechanics")
    cv2.setMouseCallback("Azure Kinect Biomechanics", mouse_callback)

    print("System ready. Click 4 points on the floor to establish the plane.")
    print("Press 'E' to start a 10-second CSV data export.")
    print("Press 'R' to remove the floor. Press 'Q' or 'ESC' to quit.")

    while True:
        # Capture frames from Azure Kinect
        capture = device.update()
        ret_color, color_image = capture.get_color_image()
        ret_depth, depth_image = capture.get_transformed_depth_image()

        if not ret_color or not ret_depth:
            continue
            
        img_h, img_w = depth_image.shape[:2]
            
        # MediaPipe needs RGB color image to detect pose properly
        image_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGRA2RGB)
        results = pose.process(image_rgb)
        
        # Prepare depth image for visualization
        # Clip max distance to 4 meters (4000mm) and convert to 8-bit
        depth_8bit = np.clip(depth_image / 4000.0 * 255.0, 0, 255).astype(np.uint8)
        display_image = cv2.applyColorMap(depth_8bit, cv2.COLORMAP_JET)
        
        current_time = time.time()
        dt = current_time - prev_time

        # ---------------------------------------------------------
        # Floor Plane Logic
        # ---------------------------------------------------------
        if len(floor_clicks) > len(floor_points_3d):
            px, py = floor_clicks[-1]
            nx, ny = px / img_w, py / img_h
            pt_3d = get_3d_point(nx, ny, depth_image)
            
            if pt_3d is not None:
                floor_points_3d.append(pt_3d)
                print(f"Point {len(floor_points_3d)} registered.")
            else:
                print(f"Warning: No depth data at the clicked point. Try clicking a slightly different spot.")
                floor_clicks.pop() 

        if len(floor_points_3d) == 4 and floor_normal is None:
            floor_normal, floor_centroid = fit_plane(np.array(floor_points_3d))
            print("Floor plane successfully initialized.")

        for pt in floor_clicks:
            cv2.circle(display_image, pt, 10, (255, 0, 255), -1)

        if floor_normal is not None:
            cv2.putText(display_image, "Floor: ACTIVE", (560, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        else:
            cv2.putText(display_image, f"Floor: PENDING ({len(floor_clicks)}/4)", (510, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)

        # ---------------------------------------------------------
        # Biomechanics Calculations
        # ---------------------------------------------------------
        # Default values to ensure variables exist even if pose is lost
        aspect_ratio = 0.0
        downward_velocity = 0.0
        angle_deg = 0.0
        fall_risk_percent = 0.0

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(display_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            lms = results.pose_landmarks.landmark

            # 1. Aspect Ratio of 3D Body Box
            valid_3d_landmarks = []
            for lm in lms:
                pt_3d = get_3d_point(lm.x, lm.y, depth_image)
                if pt_3d is not None:
                    valid_3d_landmarks.append(pt_3d)
            
            if valid_3d_landmarks:
                xs_3d = [pt[0] for pt in valid_3d_landmarks]
                ys_3d = [pt[1] for pt in valid_3d_landmarks]
                zs_3d = [pt[2] for pt in valid_3d_landmarks]
                
                dx = max(xs_3d) - min(xs_3d)
                dy = max(ys_3d) - min(ys_3d)
                dz = max(zs_3d) - min(zs_3d)
                
                if dy > 0:
                    aspect_ratio = math.sqrt(dx**2 + dz**2) / dy

            # Get 3D Points for Biomechanics (Joints)
            hip_l_3d = get_3d_point(lms[mp_pose.PoseLandmark.LEFT_HIP.value].x, lms[mp_pose.PoseLandmark.LEFT_HIP.value].y, depth_image)
            hip_r_3d = get_3d_point(lms[mp_pose.PoseLandmark.RIGHT_HIP.value].x, lms[mp_pose.PoseLandmark.RIGHT_HIP.value].y, depth_image)
            
            nose_3d = get_3d_point(lms[mp_pose.PoseLandmark.NOSE.value].x, lms[mp_pose.PoseLandmark.NOSE.value].y, depth_image)
            ankle_l_3d = get_3d_point(lms[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, lms[mp_pose.PoseLandmark.LEFT_ANKLE.value].y, depth_image)
            ankle_r_3d = get_3d_point(lms[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, lms[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y, depth_image)

            # 2. Downward Velocity of COM
            if hip_l_3d is not None and hip_r_3d is not None:
                com_3d = (hip_l_3d + hip_r_3d) / 2.0
                if prev_com_y is not None and dt > 0:
                    downward_v = ((com_3d[1] - prev_com_y) / 1000.0) / dt 
                    downward_velocity = (ema_alpha * downward_v) + ((1 - ema_alpha) * downward_velocity)
                prev_com_y = com_3d[1]

            # 3. Centerline Angle vs Floor
            if floor_normal is not None and nose_3d is not None and ankle_l_3d is not None and ankle_r_3d is not None:
                ankle_mid_3d = (ankle_l_3d + ankle_r_3d) / 2.0
                centerline_vec = ankle_mid_3d - nose_3d
                mag_vec = np.linalg.norm(centerline_vec)
                mag_norm = np.linalg.norm(floor_normal)
                
                if mag_vec > 0 and mag_norm > 0:
                    dot_prod = np.abs(np.dot(centerline_vec, floor_normal))
                    angle_rad = np.arcsin(dot_prod / (mag_vec * mag_norm))
                    angle_deg = np.degrees(angle_rad)

            # ---------------------------------------------------------
            # Fall Risk Calculation (Logistic Regression)
            # ---------------------------------------------------------
            z = B_0 + (B_1 * (downward_velocity - 0.09)) + (B_2 * (45 - angle_deg)) + (B_3 * (aspect_ratio - 1))
            z_clamped = max(min(z, 50), -50)
            fall_risk_percent = (1.0 / (1.0 + math.exp(-z_clamped))) * 100.0
            
            fall_risk_history.append(fall_risk_percent)

            risk_color = (0, 0, 255) if fall_risk_percent > 50.0 else (0, 255, 0)

            # Display Parameters
            cv2.putText(display_image, f"Realtime Fall Risk Parameters", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(display_image, f"- Downward Velocity: {downward_velocity:.2f} [m/s]", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(display_image, f"- Centerline Angle: {angle_deg:.1f} [deg]", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(display_image, f"- Aspect Ratio: {aspect_ratio:.2f}", (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            
            # Fall Risk Display
            cv2.putText(display_image, f"Fall Risk: {fall_risk_percent:.1f} [%]", (20, 220), cv2.FONT_HERSHEY_SIMPLEX, 1.0, risk_color, 3)

        # Display Instruction Message
        cv2.putText(display_image, "SHORTCUTS", (900, 30), 0, 0.4, (255, 255, 255), 1)
        cv2.putText(display_image, "- Click [Left Mouse] to create floor plane", (900, 50), 0, 0.4, (255, 255, 255), 1)
        cv2.putText(display_image, "- Press [R] to remove floor plane", (900, 70), 0, 0.4, (255, 255, 255), 1)
        cv2.putText(display_image, "- Press [E] to export parameters to CSV file", (900, 90), 0, 0.4, (255, 255, 255), 1)
        cv2.putText(display_image, "- Press [Q] to quit the program", (900, 110), 0, 0.4, (255, 255, 255), 1)

        # ---------------------------------------------------------
        # CSV Logging Logic (10 Seconds)
        # ---------------------------------------------------------
        if is_logging:
            elapsed_time = current_time - log_start_time
            if elapsed_time >= 10.0:
                # Stop logging
                is_logging = False
                if log_file:
                    log_file.close()
                    log_file = None
                print("10-second CSV data export completed.")
            else:
                # Write current frame data to CSV
                timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                csv_writer.writerow([timestamp_str, round(downward_velocity, 4), round(angle_deg, 4), round(aspect_ratio, 4), round(fall_risk_percent, 2)])
                
                # Visual Indicator on Screen
                remaining = 10.0 - elapsed_time
                cv2.putText(display_image, f"RECORDING CSV ({remaining:.1f}s)", (900, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Draw the real-time graph on the top right
        graph_width, graph_height = 360, 100
        padding = 20
        if len(fall_risk_history) > 0:
            draw_risk_graph(display_image, fall_risk_history, img_w - graph_width - padding, 120 + padding, graph_width, graph_height)

        prev_time = current_time

        cv2.imshow("Azure Kinect Biomechanics", display_image)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27: 
            break
        elif key == ord('r'):
            floor_clicks = []
            floor_points_3d = []
            floor_normal = None
            floor_centroid = None
            fall_risk_history.clear()
            print("Floor plane removed.")
        elif key == ord('e') or key == ord('E'):
            if not is_logging:
                is_logging = True
                log_start_time = time.time()
                
                # Create a uniquely named file based on current time
                filename = datetime.now().strftime("fall_risk_export_%Y%m%d_%H%M%S.csv")
                log_file = open(filename, mode='w', newline='')
                csv_writer = csv.writer(log_file)
                
                # Write header row
                csv_writer.writerow(['Timestamp', 'Downward_Velocity_m_s', 'Centerline_Angle_deg', 'Aspect_Ratio', 'Fall_Risk_percent'])
                print(f"Started 10-second CSV data export: {filename}")

    # Cleanup
    if is_logging and log_file:
        log_file.close() # Ensure file is saved properly if user exits early
        
    device.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()