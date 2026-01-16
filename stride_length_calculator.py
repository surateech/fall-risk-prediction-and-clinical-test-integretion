import cv2
import pyk4a
import mediapipe as mp
import numpy as np
import time
import statistics

def main():
    # ---------------------------------------------------------
    # 1. Azure Kinect Setup
    # ---------------------------------------------------------
    
    # Azure Kinect Setup
    config = pyk4a.Config(
        color_resolution=pyk4a.ColorResolution.RES_720P,
        depth_mode=pyk4a.DepthMode.NFOV_UNBINNED,
        camera_fps=pyk4a.FPS.FPS_30,
        synchronized_images_only=True
    )
    
    try:
        k4a = pyk4a.PyK4A(config=config)
        k4a.start()
        calibration = k4a.calibration 
    except Exception as e:
        print(f"Error starting Azure Kinect: {e}")
        return
    
    #===============================================================================

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
        mp_pose.PoseLandmark.LEFT_EYE_INNER, 
        mp_pose.PoseLandmark.LEFT_EYE, 
        mp_pose.PoseLandmark.LEFT_EYE_OUTER,
        mp_pose.PoseLandmark.RIGHT_EYE_INNER, 
        mp_pose.PoseLandmark.RIGHT_EYE, 
        mp_pose.PoseLandmark.RIGHT_EYE_OUTER,
        mp_pose.PoseLandmark.LEFT_EAR, 
        mp_pose.PoseLandmark.RIGHT_EAR,
        mp_pose.PoseLandmark.MOUTH_LEFT, 
        mp_pose.PoseLandmark.MOUTH_RIGHT
    }
    
    CUSTOM_CONNECTIONS = frozenset(
        connection for connection in mp_pose.POSE_CONNECTIONS
        if connection[0] not in EXCLUDED_LANDMARKS and connection[1] not in EXCLUDED_LANDMARKS
    )
    
    #===============================================================================

    # All functions of the process to get 3d joints
    def landmark_to_pixel(landmark, width, height):
        cx = int(landmark.x * width)
        cy = int(landmark.y * height)
        return cx, cy

    def get_depth_robust(depth_map, cx, cy, k=2):
        h, w = depth_map.shape
        xs = range(max(0, cx - k), min(w, cx + k + 1))
        ys = range(max(0, cy - k), min(h, cy + k + 1))
        depths = [
            depth_map[y, x]
            for y in ys for x in xs
            if depth_map[y, x] > 0
        ]
        if not depths:
            return None
        return float(np.median(depths))

    def pixel_to_3d(cx, cy, depth_mm, calib):
        point3d = calib.convert_2d_to_3d(
            (cx, cy),
            depth_mm,
            pyk4a.CalibrationType.COLOR
        )
        return np.array(point3d) / 1000.0

    def get_3d_pos_from_landmark(landmark, width, height, depth_map, calib, k=2):
        cx, cy = landmark_to_pixel(landmark, width, height)
        if not (0 <= cx < width and 0 <= cy < height):
            return None
        depth_mm = get_depth_robust(depth_map, cx, cy, k)
        if depth_mm is None:
            return None
        return pixel_to_3d(cx, cy, depth_mm, calib)
    
    #===============================================================================

    # ---------------------------------------------------------
    # 2. Tracking Variables
    # ---------------------------------------------------------
    
    pos_left_heel = 0.0
    pos_right_heel = 0.0
    
    pos_left_hip = 0.0
    pos_right_hip = 0.0
    pose_sacrum = (pos_left_hip + pos_right_hip) / 2
    
    current_time = 0.0
    left_heel_strike = []
    right_heel_strike = []

    left_stride_length = []
    right_stride_length = []

    LEFT_IN_CONTACT = False
    RIGHT_IN_CONTACT = False

    HEIGHT_THRESHOLD = 0.03  # meters

    #===============================================================================

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
                
                # 4. Logic
                left_heel_lm = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HEEL]
                right_heel_lm = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HEEL]
                left_hip_lm = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
                right_hip_lm = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]

                left_heel_3d = get_3d_pos_from_landmark(left_heel_lm, w_img, h_img, depth_raw, calibration)
                right_heel_3d = get_3d_pos_from_landmark(right_heel_lm, w_img, h_img, depth_raw, calibration)
                left_hip_3d = get_3d_pos_from_landmark(left_hip_lm, w_img, h_img, depth_raw, calibration)
                right_hip_3d = get_3d_pos_from_landmark(right_hip_lm, w_img, h_img, depth_raw, calibration)

                if left_heel_3d is not None:
                    lh_x, lh_y, lh_z = left_heel_3d
                if right_heel_3d is not None:
                    rh_x, rh_y, rh_z = right_heel_3d
                if left_hip_3d is not None:
                    lhp_x, _, lhp_z = left_hip_3d
                if right_hip_3d is not None:
                    rhp_x, _, rhp_z = right_hip_3d

                prev_left_y = None
                prev_right_y = None

                current_time = time.time()

                # LEFT FOOT STRIKE
                if left_heel_3d is not None:
                    if prev_left_y is not None:
                        if (lh_y < HEIGHT_THRESHOLD and
                            prev_left_y >= HEIGHT_THRESHOLD and
                            not LEFT_IN_CONTACT):

                            left_heel_strike.append({
                                "time": current_time,
                                "pos": np.array([(lh_x-lhp_x), (lh_z-lhp_z)])
                            })
                            LEFT_IN_CONTACT = True

                        if lh_y >= HEIGHT_THRESHOLD:
                            LEFT_IN_CONTACT = False

                    prev_left_y = lh_y

                # RIGHT FOOT STRIKE
                if right_heel_3d is not None:
                    if prev_right_y is not None:
                        if (rh_y < HEIGHT_THRESHOLD and
                            prev_right_y >= HEIGHT_THRESHOLD and
                            not RIGHT_IN_CONTACT):

                            right_heel_strike.append({
                                "time": current_time,
                                "pos": np.array([(rh_x-rhp_x), (rh_z-rhp_z)])
                            })
                            RIGHT_IN_CONTACT = True

                        if rh_y >= HEIGHT_THRESHOLD:
                            RIGHT_IN_CONTACT = False

                    prev_right_y = rh_y

                # LEFT STRIDE LENGTH
                if len(left_heel_strike) >= 2:
                    p1 = left_heel_strike[-2]["pos"]
                    p2 = left_heel_strike[-1]["pos"]

                    stride = np.linalg.norm(p2 - p1)
                    left_stride_length.append(stride)

                # RIGHT STRIDE LENGTH
                if len(right_heel_strike) >= 2:
                    p1 = right_heel_strike[-2]["pos"]
                    p2 = right_heel_strike[-1]["pos"]

                    stride = np.linalg.norm(p2 - p1)
                    right_stride_length.append(stride)

                #===============================================================================

                # ---------------------------------------------------------
                # 3. Display HUD (On Depth Image)
                # ---------------------------------------------------------

                # Darker background panel
                cv2.rectangle(display_image, (5, 5), (500, 400), (30, 30, 30), -1)

                # White text
                if len(left_stride_length) > 0:
                    cv2.putText(display_image, 
                    "LEFT STRIDE LENGTH: " + str(statistics.fmean(left_stride_length)), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                else:
                    cv2.putText(display_image, 
                    "LEFT STRIDE LENGTH: " + str(0), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                if len(right_stride_length) > 0:
                    cv2.putText(display_image, 
                    "RIGHT STRIDE LENGTH: " + str(statistics.fmean(right_stride_length)), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                else:
                    cv2.putText(display_image, 
                    "RIGHT STRIDE LENGTH: " + str(0), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                # Show image
                cv2.imshow("Depth Image", display_image)

            # Press 'q' to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        pass
    finally:
        k4a.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()