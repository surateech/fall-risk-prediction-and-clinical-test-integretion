import cv2
import mediapipe as mp
import numpy as np
import pyk4a
from pyk4a import Config, PyK4A, ColorResolution, DepthMode, FPS, CalibrationType
import time
import math
from collections import deque

# --- Configuration & ThresholdS ---
SITTING_THRESHOLD_Y = 0.20       # [m]
VELOCITY_THRESHOLD = 0.20        # [m/s]
PEAK_PROMINENCE = 0.15           # [m] Minimum ankle separation to count a step
MIN_TIME_BETWEEN_STEPS = 0.30    # [s]
SMOOTHING_FACTOR = 0.40          # Trust new value in probability

class GaitAnalyzer:
    def __init__(self):
        self.is_running = True
        self.reset_data()

    def reset_data(self):
        self.current_posture = "Unknown"
        self.com_velocity_avg = 0.0     
        self.velocity_buffer = deque(maxlen=5) # Smooths out spikes
        
        # Real-time Results
        self.step_count = 0
        self.current_step_length = 0.0
        self.current_step_time = 0.0
        
        self.total_step_lengths = []
        self.total_step_times = []
        
        # TUG Timer
        self.tug_state = "WAITING"       # WAITING, ACTIVE, FINISHED
        self.tug_start_time = None
        self.tug_duration = 0.0
        
        # Tracking
        self.prev_com_3d = None
        self.prev_time = None
        
        # Peak Detection
        self.ankle_dist_smooth = 0.0
        self.peak_buffer = []            
        self.last_peak_time = 0

    def calculate_3d_dist(self, p1, p2):
        return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 + (p1[2]-p2[2])**2)

    def process_pose(self, landmarks, calibration, depth_img_aligned):
        h, w = depth_img_aligned.shape
        
        def get_3d(idx):
            lm = landmarks.landmark[idx]
            cx, cy = int(lm.x * w), int(lm.y * h)
            cx, cy = max(0, min(cx, w-1)), max(0, min(cy, h-1))
            
            # Robust Depth: Sample 3x3 area to avoid 0-depth noise
            roi = depth_img_aligned[max(0, cy-1):cy+2, max(0, cx-1):cx+2]
            valid_depths = roi[roi > 0]
            if len(valid_depths) == 0: return None
            depth_val = np.median(valid_depths)
            
            try:
                point_3d = calibration.convert_2d_to_3d((cx, cy), depth_val, CalibrationType.COLOR)
                return np.array(point_3d) / 1000.0 # Convert to Meters
            except: return None

        # Get Joints
        hip_l, hip_r = get_3d(23), get_3d(24)
        knee_l, knee_r = get_3d(25), get_3d(26)
        ankle_l, ankle_r = get_3d(27), get_3d(28)

        if any(x is None for x in [hip_l, hip_r, knee_l, knee_r]):
            self.current_posture = "Lost Tracking"
            return 

        # 1. Posture & Velocity Logic
        com_3d = (hip_l + hip_r) / 2.0
        current_time = time.time()

        if self.prev_com_3d is not None:
            dt = current_time - self.prev_time
            if dt > 0:
                inst_v = self.calculate_3d_dist(com_3d, self.prev_com_3d) / dt
                if inst_v < 4.0: # Filter out tracking jumps
                    self.velocity_buffer.append(inst_v)
                    self.com_velocity_avg = sum(self.velocity_buffer) / len(self.velocity_buffer)

        self.prev_com_3d, self.prev_time = com_3d, current_time

        # Detect Posture
        diff_l, diff_r = abs(hip_l[1] - knee_l[1]), abs(hip_r[1] - knee_r[1])
        if (diff_l < SITTING_THRESHOLD_Y) and (diff_r < SITTING_THRESHOLD_Y):
            self.current_posture = "Sitting"
        elif self.com_velocity_avg > VELOCITY_THRESHOLD:
            self.current_posture = "Walking"
        else:
            self.current_posture = "Standing"

        # 2. TUG State Machine
        if self.tug_state == "WAITING" and self.current_posture != "Sitting":
            self.tug_start_time = time.time()
            self.tug_state = "ACTIVE"
        
        elif self.tug_state == "ACTIVE":
            self.tug_duration = time.time() - self.tug_start_time
            if self.current_posture == "Sitting" and self.tug_duration > 1.5:
                self.tug_state = "FINISHED"

            # 3. Step Counting (Peak Detection)
            if ankle_l is not None and ankle_r is not None:
                dist = self.calculate_3d_dist(ankle_l, ankle_r)
                self.ankle_dist_smooth = (SMOOTHING_FACTOR * dist) + ((1.0 - SMOOTHING_FACTOR) * self.ankle_dist_smooth)
                
                self.peak_buffer.append((self.ankle_dist_smooth, current_time))
                if len(self.peak_buffer) > 3: self.peak_buffer.pop(0)

                if len(self.peak_buffer) == 3:
                    p, c, n = self.peak_buffer[0][0], self.peak_buffer[1][0], self.peak_buffer[2][0]
                    if p < c and c > n and c > PEAK_PROMINENCE:
                        if (current_time - self.last_peak_time) > MIN_TIME_BETWEEN_STEPS:
                            # Update: Only count if posture is specifically "Walking"
                            if self.current_posture == "Walking":
                                self.step_count += 1
                                self.current_step_length = c
                                if self.last_peak_time > 0:
                                    self.current_step_time = current_time - self.last_peak_time
                                self.last_peak_time = current_time

def main():
    k4a = PyK4A(Config(color_resolution=ColorResolution.RES_720P, depth_mode=DepthMode.NFOV_UNBINNED, camera_fps=FPS.FPS_30, synchronized_images_only=True))
    k4a.start()
    
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)
    analyzer = GaitAnalyzer()

    while True:
        capture = k4a.get_capture()
        if capture.color is None or capture.transformed_depth is None: continue
        
        color_img = capture.color[:, :, :3].copy()
        results = pose.process(cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB))
        
        # Change view to depth with a color gradient (scaled for max 5 meters to get good contrast)
        depth_normalized = cv2.convertScaleAbs(capture.transformed_depth, alpha=255.0/5000.0)
        color_img = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
        
        if results.pose_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(color_img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            analyzer.process_pose(results.pose_landmarks, k4a.calibration, capture.transformed_depth)

        # --- UI LAYOUT ---
        h, w, _ = color_img.shape
        
        # 1. Top Left Corner: TUG & Gait
        cv2.putText(color_img, f"TUG Test Time: {analyzer.tug_duration:.2f} [s]", (20, 40), 0, 1.0, (0, 255, 255), 3)
        #cv2.putText(color_img, "Realtime Gait Parameters", (20, 120), 0, 0.8, (0, 255, 255), 2)
        #cv2.putText(color_img, f"- Step Count: {analyzer.step_count}", (20, 160), 0, 0.8, (0, 255, 255), 2)
        #cv2.putText(color_img, f"- Step Length: {analyzer.current_step_length:.2f} [m]", (20, 200), 0, 0.8, (0, 255, 255), 2)
        #cv2.putText(color_img, f"- Step Time: {analyzer.current_step_time:.2f} [s]", (20, 240), 0, 0.8, (0, 255, 255), 2)

        # 2. Top Middle: Posture
        p_text = f"Posture: {analyzer.current_posture}"
        t_size = cv2.getTextSize(p_text, 0, 0.8, 2)[0]
        cv2.putText(color_img, p_text, (int((w - t_size[0])/2), 40), 0, 0.8, (0, 255, 0), 2)

        # 3. Middle: Fall Risk Result (Show when FINISHED)
        if analyzer.tug_state == "FINISHED":
            if analyzer.tug_duration < 13.5:
                res_msg = "Test Result: LOW RISK TO FALL"
                res_color = (0, 255, 0)  # Green
            else:
                res_msg = "Test Result: HIGH RISK TO FALL"
                res_color = (0, 0, 255)  # Red
            
            # Calculate centering for the result message
            res_size = cv2.getTextSize(res_msg, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
            res_x = (w - res_size[0]) // 2
            res_y = (h + res_size[1]) // 2
            
            # Optional: Draw a dark semi-transparent background for the message
            cv2.rectangle(color_img, (res_x - 20, res_y - res_size[1] - 20), 
                          (res_x + res_size[0] + 20, res_y + 20), (0, 0, 0), -1)
            cv2.putText(color_img, res_msg, (res_x, res_y), cv2.FONT_HERSHEY_SIMPLEX, 1.2, res_color, 3)

        # 4. Bottom: Reset/Quit Message
        cv2.putText(color_img, "SHORTCUTS", (1050, 30), 0, 0.4, (255, 255, 255), 1)
        cv2.putText(color_img, "- Press [R] to test again", (1050, 50), 0, 0.4, (255, 255, 255), 1)
        cv2.putText(color_img, "- Press [Q] to quit the program", (1050, 70), 0, 0.4, (255, 255, 255), 1)

        cv2.imshow("Automated TUG Test", color_img)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('r'): analyzer.reset_data()
        elif key == ord('q'): break

    k4a.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()