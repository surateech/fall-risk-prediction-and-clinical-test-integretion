import cv2
import mediapipe as mp
import numpy as np
import pyk4a
from pyk4a import Config, PyK4A
import time

# ---------------------------------------------------------
# NEW FUNCTION: Fall Risk Assessment based on Table 2
# ---------------------------------------------------------
def assess_fall_risk(kmd, ratio, mhk):
    """
    Assess Fall Risk based on significant variables from Table 2.
    Thresholds are calculated as midpoints between 'Falls' and 'No Falls' medians.
    Significant variables: KMD (P<0.001), TMD/KMD (P=0.002), MHK (P<0.001).
    """
    score = 0
    
    # Thresholds (Midpoints)
    TH_KMD = 5.06    # (4.812 + 5.309) / 2
    TH_RATIO = 0.136 # (0.145 + 0.127) / 2
    TH_MHK = 0.082   # (0.073 + 0.091) / 2
    
    # 1. KMD Logic: Lower KMD indicates Higher Risk (Falls group median 4.812 vs 5.309)
    if kmd < TH_KMD:
        score += 1
        
    # 2. Ratio Logic: Higher Ratio indicates Higher Risk (Falls group median 0.145 vs 0.127)
    if ratio > TH_RATIO:
        score += 1
        
    # 3. MHK Logic: Lower Step Height indicates Higher Risk (Falls group median 0.073 vs 0.091)
    if mhk < TH_MHK:
        score += 1
        
    # Decision: If 2 or more indicators point to risk, classify as High Fall
    if score >= 2:
        return "High Fall"
    else:
        return "Low Fall"

# Helper function to calculate mean safely
def calculate_mean(data_list):
    if not data_list:
        return 0.0
    return sum(data_list) / len(data_list)

def main():
    
    # ---------------------------------------------------------
    # 1. Configuration & Setup
    # ---------------------------------------------------------
    WARMUP_DURATION = 10    
    RECORD_DURATION = 10    
    TOTAL_DURATION = WARMUP_DURATION + RECORD_DURATION
    
    config = Config(
        color_resolution=pyk4a.ColorResolution.RES_720P,
        depth_mode=pyk4a.DepthMode.NFOV_UNBINNED, 
        camera_fps=pyk4a.FPS.FPS_30,
        synchronized_images_only=True,
    )
    
    try:
        k4a = PyK4A(config=config)
        k4a.start()
        calibration = k4a.calibration 
    except Exception as e:
        print(f"Error starting Azure Kinect: {e}")
        return

    # MediaPipe Setup
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1, 
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    mp_drawing = mp.solutions.drawing_utils

    EXCLUDED_LANDMARKS = {
        mp_pose.PoseLandmark.LEFT_EYE_INNER, mp_pose.PoseLandmark.LEFT_EYE, mp_pose.PoseLandmark.LEFT_EYE_OUTER,
        mp_pose.PoseLandmark.RIGHT_EYE_INNER, mp_pose.PoseLandmark.RIGHT_EYE, mp_pose.PoseLandmark.RIGHT_EYE_OUTER,
        mp_pose.PoseLandmark.LEFT_EAR, mp_pose.PoseLandmark.RIGHT_EAR,
        mp_pose.PoseLandmark.MOUTH_LEFT, mp_pose.PoseLandmark.MOUTH_RIGHT
    }

    CUSTOM_CONNECTIONS = frozenset(
        connection for connection in mp_pose.POSE_CONNECTIONS
        if connection[0] not in EXCLUDED_LANDMARKS and connection[1] not in EXCLUDED_LANDMARKS
    )

    def get_3d_pos(landmark, width, height, depth_map, calib):
        cx, cy = int(landmark.x * width), int(landmark.y * height)
        if 0 <= cx < width and 0 <= cy < height:
            d_mm = depth_map[cy, cx]
            if d_mm > 0:
                point3d = calib.convert_2d_to_3d((cx, cy), d_mm, pyk4a.CalibrationType.COLOR)
                return np.array(point3d) / 1000.0
        return None

    # ---------------------------------------------------------
    # 2. Tracking Variables
    # ---------------------------------------------------------
    
    test_start_time = None
    is_recording = False  
    test_finished = False # New Flag
    
    H0_ref = None           
    max_movement_dist = 0.0 
    TMD_sum = 0.0           
    head_pos_at_last_second = None
    
    prev_K_L = None 
    prev_K_R = None
    K_sum_L = 0.0
    K_sum_R = 0.0

    step_L_count = 0
    step_R_count = 0
    total_steps = 0
    init_K_y_L = None 
    init_K_y_R = None
    is_L_leg_up = False
    is_R_leg_up = False
    STEP_LIFT_THRESHOLD = 0.015 

    temp_step_L_max = 0.0
    temp_step_L_min = 0.0
    temp_step_R_max = 0.0
    temp_step_R_min = 0.0

    step_height_diffs_L = [] 
    step_height_diffs_R = []
    
    # Final Result Storage
    final_results = {
        "MMD": 0.0, "TMD": 0.0, "KMD": 0.0, 
        "Ratio": 0.0, "MHK": 0.0, "Steps": 0,
        "Risk": "Evaluating..."
    }

    font = cv2.FONT_HERSHEY_SIMPLEX

    print("--------------------------------------------------")
    print("Mode: DEPTH VISUALIZATION ACTIVATED")
    print(f"Phase 1: {WARMUP_DURATION}s Warmup.")
    print(f"Phase 2: {RECORD_DURATION}s Data Collection.")
    print("--------------------------------------------------")

    try:
        while True:
            capture = k4a.get_capture()
            
            if capture.color is not None and capture.transformed_depth is not None:
                
                # 1. Prepare RGB for MediaPipe Inference
                img_rgb = cv2.cvtColor(capture.color[:, :, :3], cv2.COLOR_BGR2RGB)
                
                # Only process pose if test is NOT finished to save resources, 
                # or continue processing for visualization but stop calculations
                results = pose.process(img_rgb)
                
                h_img, w_img, _ = capture.color.shape

                # 2. Prepare DEPTH for Visualization
                depth_raw = capture.transformed_depth
                depth_vis = cv2.convertScaleAbs(depth_raw, alpha=0.05)
                display_image = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

                # 3. Draw Landmarks
                if results.pose_landmarks:
                    mp_drawing.draw_landmarks(
                        display_image,
                        results.pose_landmarks, 
                        connections=CUSTOM_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255,255,255), thickness=2, circle_radius=2),
                        connection_drawing_spec=mp_drawing.DrawingSpec(color=(200,200,200), thickness=2)
                    )

                # =========================================================
                # 4. LOGIC
                # =========================================================
                
                current_phase_time = 0.0
                phase_label = "WAITING"
                
                if test_finished:
                    phase_label = "FINISHED"
                
                elif test_start_time is None and results.pose_landmarks:
                    # Start Trigger
                    nose_lm = results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]
                    cx, cy = int(nose_lm.x * w_img), int(nose_lm.y * h_img)
                    if 0 <= cx < w_img and 0 <= cy < h_img:
                         if capture.transformed_depth[cy, cx] > 0:
                             test_start_time = time.time()
                             print("User Detected. Warmup Started.")

                elif test_start_time is not None:
                    elapsed_total = time.time() - test_start_time
                    
                    if elapsed_total < WARMUP_DURATION:
                        is_recording = False
                        current_phase_time = WARMUP_DURATION - elapsed_total
                        phase_label = "WARMUP"
                    
                    elif elapsed_total < TOTAL_DURATION:
                        if not is_recording: 
                            is_recording = True
                            print("=== Warmup Done. RESETTING DATA. Recording Started. ===")
                            
                            # Reset Logic
                            max_movement_dist = 0.0
                            TMD_sum = 0.0
                            K_sum_L = 0.0
                            K_sum_R = 0.0
                            step_L_count = 0
                            step_R_count = 0
                            step_height_diffs_L = []
                            step_height_diffs_R = []
                            
                            if results.pose_landmarks:
                                nose_lm = results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]
                                curr_H_3d = get_3d_pos(nose_lm, w_img, h_img, capture.transformed_depth, calibration)
                                
                                cx, cy = int(nose_lm.x * w_img), int(nose_lm.y * h_img)
                                if 0 <= cx < w_img and 0 <= cy < h_img:
                                    d = capture.transformed_depth[cy, cx]
                                    if d > 0: H0_ref = d / 1000.0
                                
                                head_pos_at_last_second = curr_H_3d
                                
                                lm_L = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE]
                                lm_R = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE]
                                kL_3d = get_3d_pos(lm_L, w_img, h_img, capture.transformed_depth, calibration)
                                kR_3d = get_3d_pos(lm_R, w_img, h_img, capture.transformed_depth, calibration)
                                prev_K_L = kL_3d
                                prev_K_R = kR_3d
                                
                                if kL_3d is not None: init_K_y_L = kL_3d[1]
                                if kR_3d is not None: init_K_y_R = kR_3d[1]
                                is_L_leg_up = False
                                is_R_leg_up = False

                        current_phase_time = TOTAL_DURATION - elapsed_total
                        phase_label = "RECORDING"
                    else:
                        # === TIME UP: CALCULATE RESULTS ONCE ===
                        is_recording = False
                        test_finished = True
                        phase_label = "FINISHED"
                        
                        kh_r_final = calculate_mean(step_height_diffs_R)
                        kh_l_final = calculate_mean(step_height_diffs_L)
                        mkh_final = (kh_r_final + kh_l_final) / 2.0
                        kmd_final = (K_sum_L + K_sum_R) / 2.0
                        
                        ratio_val = 0.0
                        if kmd_final > 0:
                            ratio_val = TMD_sum / kmd_final
                            
                        risk_level = assess_fall_risk(kmd_final, ratio_val, mkh_final)
                        
                        final_results["MMD"] = max_movement_dist
                        final_results["TMD"] = TMD_sum
                        final_results["KMD"] = kmd_final
                        final_results["Ratio"] = ratio_val
                        final_results["MHK"] = mkh_final
                        final_results["Steps"] = step_L_count + step_R_count
                        final_results["Risk"] = risk_level
                        
                        print("\n=== Test Finished ===")
                        print(f"Risk Assessment: {risk_level}")

                # ----------------------------------------------------------------
                # Calculations (Only if recording and not finished)
                # ----------------------------------------------------------------
                if is_recording and not test_finished and results.pose_landmarks:
                    nose_lm = results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]
                    curr_H = get_3d_pos(nose_lm, w_img, h_img, capture.transformed_depth, calibration)
                    
                    cx_nose, cy_nose = int(nose_lm.x * w_img), int(nose_lm.y * h_img)
                    if 0 <= cx_nose < w_img and 0 <= cy_nose < h_img:
                        d_mm = capture.transformed_depth[cy_nose, cx_nose]
                        if d_mm > 0:
                            curr_depth = d_mm / 1000.0
                            if H0_ref is not None:
                                delta = abs(curr_depth - H0_ref)
                                if delta > max_movement_dist: max_movement_dist = delta
                            else: H0_ref = curr_depth

                    if curr_H is not None and head_pos_at_last_second is not None:
                        curr_t = time.time()
                        if (curr_t - 1.0) >= 1.0: # Note: This check relies on loop speed, might need last_second timestamp correction in real usage, but kept as is from original logic structure
                             pass 
                        # Simple distance accumulation every frame for now, or 1 sec interval as originally intended?
                        # Original logic seemed to rely on 1s interval for TMD accumulation? 
                        # To keep it responsive, I will accumulate frame-to-frame distance or stick to the snippet provided.
                        # Assuming snippet logic: "if (curr_t - 1.0) >= 1.0" is likely broken in original snippet logic (comparing current time to float 1.0). 
                        # I will FIX the TMD calculation to be frame-to-frame distance accumulation for smoother result, 
                        # OR stick to valid time comparison. Let's do frame-to-frame for accuracy.
                        
                        dist_frame = np.linalg.norm(curr_H - head_pos_at_last_second)
                        TMD_sum += dist_frame
                        head_pos_at_last_second = curr_H

                    lm_L = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE]
                    lm_R = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE]
                    curr_K_L = get_3d_pos(lm_L, w_img, h_img, capture.transformed_depth, calibration)
                    curr_K_R = get_3d_pos(lm_R, w_img, h_img, capture.transformed_depth, calibration)

                    if curr_K_L is not None:
                        if prev_K_L is not None: K_sum_L += np.linalg.norm(curr_K_L - prev_K_L)
                        prev_K_L = curr_K_L
                    if curr_K_R is not None:
                        if prev_K_R is not None: K_sum_R += np.linalg.norm(curr_K_R - prev_K_R)
                        prev_K_R = curr_K_R

                    # Step Logic
                    if curr_K_L is not None and init_K_y_L is not None:
                        height_diff_L = init_K_y_L - curr_K_L[1]
                        if not is_L_leg_up:
                            if height_diff_L > STEP_LIFT_THRESHOLD:
                                is_L_leg_up = True
                                temp_step_L_max = height_diff_L
                                temp_step_L_min = height_diff_L
                        else:
                            temp_step_L_max = max(temp_step_L_max, height_diff_L)
                            temp_step_L_min = min(temp_step_L_min, height_diff_L)
                            if height_diff_L < (STEP_LIFT_THRESHOLD / 2.0):
                                is_L_leg_up = False
                                step_L_count += 1
                                step_height_diffs_L.append(temp_step_L_max - temp_step_L_min)

                    if curr_K_R is not None and init_K_y_R is not None:
                        height_diff_R = init_K_y_R - curr_K_R[1]
                        if not is_R_leg_up:
                            if height_diff_R > STEP_LIFT_THRESHOLD:
                                is_R_leg_up = True
                                temp_step_R_max = height_diff_R
                                temp_step_R_min = height_diff_R
                        else:
                            temp_step_R_max = max(temp_step_R_max, height_diff_R)
                            temp_step_R_min = min(temp_step_R_min, height_diff_R)
                            if height_diff_R < (STEP_LIFT_THRESHOLD / 2.0):
                                is_R_leg_up = False
                                step_R_count += 1
                                step_height_diffs_R.append(temp_step_R_max - temp_step_R_min)
                                
                    total_steps = step_L_count + step_R_count

                # -----------------------------------------------------
                # Display HUD (On Depth Image)
                # -----------------------------------------------------
                
                # Darker background panel
                cv2.rectangle(display_image, (5, 5), (500, 400), (30, 30, 30), -1)

                if phase_label == "FINISHED":
                    # === RESULT SCREEN ===
                    header_color = (0, 255, 0) # Green for finished
                    cv2.putText(display_image, "TEST COMPLETED", (15, 35), font, 1.0, header_color, 2)
                    
                    # Display Fall Risk Result prominently
                    risk_color = (0, 0, 255) if final_results["Risk"] == "High Fall" else (0, 255, 255)
                    cv2.putText(display_image, f"RESULT: {final_results['Risk']}", (15, 80), font, 1.2, risk_color, 3)
                    
                    # Display Stats
                    txt_color = (200, 200, 200)
                    cv2.putText(display_image, f"TMD: {final_results['TMD']:.3f} m", (15, 130), font, 0.7, txt_color, 2)
                    cv2.putText(display_image, f"KMD: {final_results['KMD']:.3f} m", (15, 160), font, 0.7, txt_color, 2)
                    cv2.putText(display_image, f"MMD: {final_results['MMD']:.3f} m", (15, 190), font, 0.7, txt_color, 2)
                    cv2.putText(display_image, f"Ratio (TMD/KMD): {final_results['Ratio']:.3f}", (15, 220), font, 0.7, txt_color, 2)
                    cv2.putText(display_image, f"MHK (Step H): {final_results['MHK']:.3f} m", (15, 250), font, 0.7, txt_color, 2)
                    cv2.putText(display_image, f"Total Steps: {final_results['Steps']}", (15, 280), font, 0.7, txt_color, 2)
                    
                    cv2.putText(display_image, "Press 'q' to exit", (15, 350), font, 0.6, (100,100,100), 1)

                else:
                    # === RUNNING SCREEN ===
                    if phase_label == "WARMUP":
                        header_color = (0, 255, 255) 
                        header_text = f"WARMUP: {current_phase_time:.1f} s"
                    elif phase_label == "RECORDING":
                        header_color = (0, 0, 255) 
                        header_text = f"RECORDING: {current_phase_time:.1f} s"
                    else:
                        header_color = (200, 200, 200)
                        header_text = "WAITING FOR USER..."
                    
                    cv2.putText(display_image, header_text, (15, 35), font, 1.0, header_color, 2)
                    
                    dc = (0, 165, 255) if is_recording else (100, 100, 100)
                    kc = (50, 255, 50) if is_recording else (100, 100, 100)
                    sc = (255, 255, 0) if is_recording else (100, 100, 100)
                    
                    kmd_live = (K_sum_L + K_sum_R) / 2.0
                    curr_kh_l = calculate_mean(step_height_diffs_L)
                    curr_kh_r = calculate_mean(step_height_diffs_R)

                    cv2.putText(display_image, f"Head MMD: {max_movement_dist:.4f} m", (15, 70), font, 0.7, dc, 2)
                    cv2.putText(display_image, f"Head TMD: {TMD_sum:.4f} m", (15, 100), font, 0.7, dc, 2)
                    
                    cv2.putText(display_image, f"KMD: {kmd_live:.4f} m", (15, 150), font, 0.8, kc, 2)

                    cv2.putText(display_image, f"Steps L: {step_L_count} | R: {step_R_count}", (15, 200), font, 0.7, sc, 2)
                    cv2.putText(display_image, f"Avg KH L: {curr_kh_l:.3f} m", (15, 230), font, 0.6, sc, 1)
                    cv2.putText(display_image, f"Avg KH R: {curr_kh_r:.3f} m", (15, 260), font, 0.6, sc, 1)
                
                cv2.imshow('Kinect Depth Mode Analysis', display_image)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        pass
    finally:
        k4a.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()