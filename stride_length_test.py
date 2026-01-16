import cv2
import pyk4a
import mediapipe as mp
import numpy as np
import time
import statistics

def main():
    # ---------------------------------------------------------
    # 1. SETUP
    # ---------------------------------------------------------
    config = pyk4a.Config(
        color_resolution=pyk4a.ColorResolution.RES_720P,
        depth_mode=pyk4a.DepthMode.NFOV_UNBINNED,
        camera_fps=pyk4a.FPS.FPS_30,
        synchronized_images_only=True
    )
    
    k4a = pyk4a.PyK4A(config=config)
    k4a.start()
    calibration = k4a.calibration 

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1, 
        smooth_landmarks=True
    )
    mp_drawing = mp.solutions.drawing_utils

    # Helper Functions
    def landmark_to_pixel(landmark, width, height):
        cx = int(landmark.x * width)
        cy = int(landmark.y * height)
        return cx, cy

    def get_depth_robust(depth_map, cx, cy, k=2):
        h, w = depth_map.shape
        xs = range(max(0, cx - k), min(w, cx + k + 1))
        ys = range(max(0, cy - k), min(h, cy + k + 1))
        depths = [depth_map[y, x] for y in ys for x in xs if depth_map[y, x] > 0]
        if not depths: return None
        return float(np.median(depths))

    def pixel_to_3d(cx, cy, depth_mm, calib):
        point3d = calib.convert_2d_to_3d((cx, cy), depth_mm, pyk4a.CalibrationType.COLOR)
        return np.array(point3d) / 1000.0 # Convert to meters

    def get_3d_pos_from_landmark(landmark, width, height, depth_map, calib):
        cx, cy = landmark_to_pixel(landmark, width, height)
        # Check bounds
        if not (0 <= cx < width and 0 <= cy < height): return None, (cx, cy)
        depth_mm = get_depth_robust(depth_map, cx, cy, k=2)
        if depth_mm is None: return None, (cx, cy)
        return pixel_to_3d(cx, cy, depth_mm, calib), (cx, cy)

    # ---------------------------------------------------------
    # 2. VARIABLES (อยู่นอก Loop!)
    # ---------------------------------------------------------
    left_heel_strike = []
    right_heel_strike = []
    left_stride_length = []
    right_stride_length = []

    LEFT_IN_CONTACT = False
    RIGHT_IN_CONTACT = False
    
    # ปรับค่านี้: ถ้าระยะห่าง (Y) ระหว่างสะโพกกับส้นเท้า มากกว่าค่านี้ ให้ถือว่า "แตะพื้น"
    # ลองเริ่มที่ 0.25 - 0.3 เมตร ดูครับ (ค่าน้อย = Sensitive มาก, ค่ามาก = ต้องเหยียดขายาวจริงๆ ถึงจะติด)
    VERTICAL_DIST_THRESHOLD = 0.30 

    try:
        while True:
            capture = k4a.get_capture()
            if capture.color is not None and capture.transformed_depth is not None:
                
                img_rgb = cv2.cvtColor(capture.color[:, :, :3], cv2.COLOR_BGR2RGB)
                results = pose.process(img_rgb)
                
                # Visualization Setup
                depth_raw = capture.transformed_depth
                depth_vis = cv2.convertScaleAbs(depth_raw, alpha=0.05)
                display_image = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
                h_img, w_img = depth_raw.shape

                current_time = time.time()

                if results.pose_landmarks:
                    # Draw Skeleton
                    mp_drawing.draw_landmarks(display_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                    
                    landmarks = results.pose_landmarks.landmark
                    
                    # --- LEFT SIDE PROCESSING ---
                    l_heel_3d, l_heel_px = get_3d_pos_from_landmark(landmarks[mp_pose.PoseLandmark.LEFT_HEEL], w_img, h_img, depth_raw, calibration)
                    l_hip_3d, _ = get_3d_pos_from_landmark(landmarks[mp_pose.PoseLandmark.LEFT_HIP], w_img, h_img, depth_raw, calibration)

                    dist_l = 0.0
                    if l_heel_3d is not None and l_hip_3d is not None:
                        # Logic: Y difference (Heel Y - Hip Y)
                        # ใน Camera Space ปกติ Y ชี้ลง (Down), ดังนั้น Heel อยู่ต่ำกว่า Hip ค่า Y จะมากกว่า
                        dist_l = l_heel_3d[1] - l_hip_3d[1] 
                        
                        is_touching_l = dist_l > VERTICAL_DIST_THRESHOLD
                        
                        # State Machine
                        if is_touching_l:
                            color = (0, 255, 0) # GREEN = Contact
                            if not LEFT_IN_CONTACT:
                                # Just landed
                                LEFT_IN_CONTACT = True
                                left_heel_strike.append({"pos": l_heel_3d[[0, 2]]}) # Store X, Z
                                if len(left_heel_strike) >= 2:
                                    stride = np.linalg.norm(left_heel_strike[-1]["pos"] - left_heel_strike[-2]["pos"])
                                    # Filter noise (stride must be realistic)
                                    if 0.1 < stride < 1.5: 
                                        left_stride_length.append(stride)
                        else:
                            color = (0, 0, 255) # RED = Swing
                            LEFT_IN_CONTACT = False
                        
                        # DEBUG DRAWING (วงกลมที่ส้นเท้า)
                        cv2.circle(display_image, l_heel_px, 10, color, -1)
                        cv2.putText(display_image, f"{dist_l:.2f}m", (l_heel_px[0]+15, l_heel_px[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

                    # --- RIGHT SIDE PROCESSING ---
                    r_heel_3d, r_heel_px = get_3d_pos_from_landmark(landmarks[mp_pose.PoseLandmark.RIGHT_HEEL], w_img, h_img, depth_raw, calibration)
                    r_hip_3d, _ = get_3d_pos_from_landmark(landmarks[mp_pose.PoseLandmark.RIGHT_HIP], w_img, h_img, depth_raw, calibration)

                    dist_r = 0.0
                    if r_heel_3d is not None and r_hip_3d is not None:
                        dist_r = r_heel_3d[1] - r_hip_3d[1]
                        is_touching_r = dist_r > VERTICAL_DIST_THRESHOLD
                        
                        if is_touching_r:
                            color = (0, 255, 0)
                            if not RIGHT_IN_CONTACT:
                                RIGHT_IN_CONTACT = True
                                right_heel_strike.append({"pos": r_heel_3d[[0, 2]]})
                                if len(right_heel_strike) >= 2:
                                    stride = np.linalg.norm(right_heel_strike[-1]["pos"] - right_heel_strike[-2]["pos"])
                                    if 0.1 < stride < 1.5:
                                        right_stride_length.append(stride)
                        else:
                            color = (0, 0, 255)
                            RIGHT_IN_CONTACT = False
                            
                        cv2.circle(display_image, r_heel_px, 10, color, -1)
                        cv2.putText(display_image, f"{dist_r:.2f}m", (r_heel_px[0]+15, r_heel_px[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

                # ---------------------------------------------------------
                # 3. HUD DISPLAY
                # ---------------------------------------------------------
                avg_l = statistics.mean(left_stride_length) if left_stride_length else 0.0
                avg_r = statistics.mean(right_stride_length) if right_stride_length else 0.0

                # Panel
                cv2.rectangle(display_image, (10, 10), (350, 120), (0,0,0), -1)
                
                # Text
                cv2.putText(display_image, f"Threshold: > {VERTICAL_DIST_THRESHOLD}m", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
                cv2.putText(display_image, f"L Stride: {avg_l:.3f} m ({len(left_stride_length)} steps)", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                cv2.putText(display_image, f"R Stride: {avg_r:.3f} m ({len(right_stride_length)} steps)", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

                cv2.imshow("Debug Mode", display_image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        pass
    finally:
        k4a.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()