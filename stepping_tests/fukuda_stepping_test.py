import cv2
import numpy as np
import mediapipe as mp
import time
import math
from pyk4a import Config, PyK4A, ColorResolution, DepthMode, CalibrationType

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

class FST_MediaPipe_System:
    def __init__(self):
        # Azure Kinect Configuration
        self.kinect = PyK4A(Config(
            color_resolution=ColorResolution.RES_720P,
            depth_mode=DepthMode.NFOV_UNBINNED,
            synchronized_images_only=True,
        ))
        
        # System States
        self.display_mode = "DEPTH" 
        self.state = "WAITING_F1"  
        self.step_count = 0
        self.target_steps = 50
        self.start_time = 0
        self.duration = 0
        self.warmup_duration = 10.0 # 10-second warmup phase
        self.latest_depth = None    
        
        # Floor Plane Data
        self.floor_pts_3d = []
        self.floor_pts_2d = []
        self.floor_normal = None
        self.floor_d = 0.0

        # Baseline Direction Data 
        self.pos_point_1_3d = None
        self.pos_point_1_2d = None
        self.pos_point_2_3d = None
        self.pos_point_2_2d = None
        
        self.final_position_3d = None
        self.current_position_3d = np.array([0.0, 0.0, 0.0])
        self.current_position_2d = None
        
        # Real-time metrics
        self.current_angle = 0.0
        self.final_angle = 0.0
        self.l_ankle_height = 0.0
        self.r_ankle_height = 0.0
        
        # Step counting heuristics
        self.left_ankle_raised = False
        self.right_ankle_raised = False

        # UI Button dimensions [x1, y1, x2, y2]
        self.btn_toggle = [10, 10, 200, 50]
        self.btn_restart = [10, 60, 200, 100]       # Soft Restart 
        self.btn_hard_reset = [10, 110, 200, 150]   # Hard Reset 

    def start(self):
        self.kinect.start()
        cv2.namedWindow("FST System")
        cv2.setMouseCallback("FST System", self.mouse_callback)
        print("System started. Camera warming up...")

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # 1. Toggle Mode Button
            if self.btn_toggle[0] <= x <= self.btn_toggle[2] and self.btn_toggle[1] <= y <= self.btn_toggle[3]:
                self.display_mode = "RGB" if self.display_mode == "DEPTH" else "DEPTH"
                return

            # 2. Hard Reset Button 
            if self.btn_hard_reset[0] <= x <= self.btn_hard_reset[2] and self.btn_hard_reset[1] <= y <= self.btn_hard_reset[3]:
                self.hard_reset_system()
                return

            # 3. Soft Restart Button 
            if self.state == "FINISHED":
                if self.btn_restart[0] <= x <= self.btn_restart[2] and self.btn_restart[1] <= y <= self.btn_restart[3]:
                    self.soft_restart_system()
                    return

            # 4. Handle Plotting
            if self.latest_depth is not None:
                depth_val = self.latest_depth[y, x]
                
                if depth_val > 0:
                    pt_3d = self.kinect.calibration.convert_2d_to_3d(
                        (x, y), depth_val, CalibrationType.COLOR, CalibrationType.COLOR
                    )
                    
                    # FLOOR CALIBRATION PHASE
                    if self.state == "WAITING_F1":
                        self.floor_pts_3d.append(np.array(pt_3d))
                        self.floor_pts_2d.append((x, y))
                        self.state = "WAITING_F2"
                    elif self.state == "WAITING_F2":
                        self.floor_pts_3d.append(np.array(pt_3d))
                        self.floor_pts_2d.append((x, y))
                        self.state = "WAITING_F3"
                    elif self.state == "WAITING_F3":
                        self.floor_pts_3d.append(np.array(pt_3d))
                        self.floor_pts_2d.append((x, y))
                        self.state = "WAITING_F4"
                    elif self.state == "WAITING_F4":
                        self.floor_pts_3d.append(np.array(pt_3d))
                        self.floor_pts_2d.append((x, y))
                        self.calculate_floor_plane()
                        self.state = "WAITING_P1"
                        
                    # BASELINE PLOTTING PHASE
                    elif self.state == "WAITING_P1":
                        self.pos_point_1_3d = np.array(pt_3d)
                        self.pos_point_1_2d = (x, y)
                        self.state = "WAITING_P2"
                    elif self.state == "WAITING_P2":
                        self.pos_point_2_3d = np.array(pt_3d)
                        self.pos_point_2_2d = (x, y)
                        
                        # Start Warm-Up Phase
                        self.state = "WARMUP"
                        self.start_time = time.time()
                        self.step_count = 0
                        print(f"Warm-Up started. Test will begin in {self.warmup_duration} seconds.")

    def soft_restart_system(self):
        """Restarts the test (going back to WARMUP) while maintaining the floor and baseline points."""
        self.state = "WARMUP" 
        self.step_count = 0
        self.start_time = time.time()
        self.duration = 0
        self.current_angle = 0.0
        self.final_angle = 0.0
        self.final_position_3d = None
        self.left_ankle_raised = False
        self.right_ankle_raised = False
        print(f"Test Restarted! Warm-Up started for {self.warmup_duration} seconds.")

    def hard_reset_system(self):
        """Clears everything including calibration to start fresh."""
        self.state = "WAITING_F1"
        self.step_count = 0
        self.floor_pts_3d = []
        self.floor_pts_2d = []
        self.floor_normal = None
        self.pos_point_1_3d = None
        self.pos_point_1_2d = None
        self.pos_point_2_3d = None
        self.pos_point_2_2d = None
        self.final_position_3d = None
        self.current_position_3d = np.array([0.0, 0.0, 0.0])
        self.duration = 0
        self.current_angle = 0.0
        self.final_angle = 0.0
        self.left_ankle_raised = False
        self.right_ankle_raised = False
        print("System fully reset. Waiting for Floor Point 1.")

    def calculate_floor_plane(self):
        """Calculates floor plane using 4 points (cross product of diagonals)."""
        if len(self.floor_pts_3d) == 4:
            A, B, C, D = self.floor_pts_3d[0], self.floor_pts_3d[1], self.floor_pts_3d[2], self.floor_pts_3d[3]
            
            vec_AC = C - A
            vec_BD = D - B
            normal = np.cross(vec_AC, vec_BD)
            mag = np.linalg.norm(normal)
            
            if mag > 1e-6:
                self.floor_normal = normal / mag
                d_vals = [-np.dot(self.floor_normal, pt) for pt in self.floor_pts_3d]
                self.floor_d = np.mean(d_vals)
                print("4-Point Floor plane established successfully.")
            else:
                print("Error: Points are invalid. Hard Restart recommended.")

    def calculate_height_to_floor(self, pt_3d):
        if self.floor_normal is None:
            return pt_3d[1] 
        return np.abs(np.dot(self.floor_normal, pt_3d) + self.floor_d)

    def get_3d_point(self, landmark, color_img, transformed_depth):
        h, w, _ = color_img.shape
        px, py = int(landmark.x * w), int(landmark.y * h)
        if 0 <= px < w and 0 <= py < h:
            depth_val = transformed_depth[py, px]
            if depth_val > 0:
                point_3d = self.kinect.calibration.convert_2d_to_3d(
                    (px, py), depth_val, CalibrationType.COLOR, CalibrationType.COLOR
                )
                return np.array(point_3d)
        return None

    def calculate_custom_angle(self, pos_point_1, pos_point_2, middle_point):
        centerline_vec = middle_point - pos_point_1
        floor_normal = pos_point_2 - pos_point_1
        
        mag_vec = np.linalg.norm(centerline_vec)
        mag_norm = np.linalg.norm(floor_normal)
        
        if mag_vec == 0 or mag_norm == 0:
            return 0.0
            
        dot_prod = np.abs(np.dot(centerline_vec, floor_normal))
        val = np.arccos(dot_prod / (mag_vec * mag_norm))
        return np.degrees(val)

    def count_steps(self):
        """Step counting logic based on calculated 3D ankle heights from the floor."""
        step_threshold = 60.0  
        
        if self.l_ankle_height > step_threshold and not self.left_ankle_raised:
            self.left_ankle_raised = True
            self.step_count += 1
        elif self.l_ankle_height <= step_threshold:
            self.left_ankle_raised = False

        if self.r_ankle_height > step_threshold and not self.right_ankle_raised:
            self.right_ankle_raised = True
            self.step_count += 1
        elif self.r_ankle_height <= step_threshold:
            self.right_ankle_raised = False

    def draw_ui(self, frame):
        # Draw Buttons
        cv2.rectangle(frame, (self.btn_toggle[0], self.btn_toggle[1]), 
                     (self.btn_toggle[2], self.btn_toggle[3]), (200, 200, 200), -1)
        cv2.putText(frame, f"MODE: {self.display_mode}", (15, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        if self.state == "FINISHED":
            cv2.rectangle(frame, (self.btn_restart[0], self.btn_restart[1]), 
                         (self.btn_restart[2], self.btn_restart[3]), (150, 255, 150), -1)
            cv2.putText(frame, "RESTART TEST", (25, 90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        cv2.rectangle(frame, (self.btn_hard_reset[0], self.btn_hard_reset[1]), 
                     (self.btn_hard_reset[2], self.btn_hard_reset[3]), (150, 150, 255), -1)
        cv2.putText(frame, "CLEAR ALL", (40, 140), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        # Draw Tracking Data 
        y_offset = 180 
        
        pos_text = f"Pos: ({self.current_position_3d[0]:.0f}, {self.current_position_3d[1]:.0f}, {self.current_position_3d[2]:.0f})mm"
        cv2.putText(frame, pos_text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        ankle_text = f"Ankle Ht (Floor) -> L: {self.l_ankle_height:.0f}mm | R: {self.r_ankle_height:.0f}mm"
        cv2.putText(frame, ankle_text, (10, y_offset + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)

        cv2.putText(frame, f"STATE: {self.state}", (10, y_offset + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f"STEPS: {self.step_count} / {self.target_steps}", (10, y_offset + 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # Dynamic Timing and Angle Display
        if self.state == "WARMUP":
            remain = max(0, self.warmup_duration - (time.time() - self.start_time))
            cv2.putText(frame, f"WARM-UP: {remain:.1f}s remaining", (10, y_offset + 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            cv2.putText(frame, "Please step in place.", (10, y_offset + 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        elif self.state == "TESTING":
            cv2.putText(frame, f"Test Time: {time.time() - self.start_time:.1f}s", (10, y_offset + 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Angle: {self.current_angle:.2f} deg", (10, y_offset + 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        elif self.state == "FINISHED":
            cv2.putText(frame, f"Final Time: {self.duration:.1f}s", (10, y_offset + 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"FINAL ANGLE: {self.final_angle:.2f} deg", (10, y_offset + 160), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)

        # Plotting Status Text
        status_y = 380
        if "WAITING_F" in self.state:
            step = self.state[-1]
            cv2.putText(frame, f"Plot Floor ({step}/4) - Click on floor", (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        elif self.state == "WAITING_P1":
            cv2.putText(frame, "Plot Point (0/2) - Click anywhere", (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        elif self.state == "WAITING_P2":
            cv2.putText(frame, "Plot Point (1/2) - Click anywhere", (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
        else:
            cv2.putText(frame, "All Points Plotted", (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Draw Floor Points & Area (Cyan)
        for i, pt in enumerate(self.floor_pts_2d):
            cv2.circle(frame, pt, 8, (255, 255, 0), -1)
            cv2.putText(frame, f"F{i+1}", (pt[0]+10, pt[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            
        if len(self.floor_pts_2d) > 1:
            cv2.line(frame, self.floor_pts_2d[0], self.floor_pts_2d[1], (255, 255, 0), 1)
        if len(self.floor_pts_2d) > 2:
            cv2.line(frame, self.floor_pts_2d[1], self.floor_pts_2d[2], (255, 255, 0), 1)
        if len(self.floor_pts_2d) == 4:
            cv2.line(frame, self.floor_pts_2d[2], self.floor_pts_2d[3], (255, 255, 0), 1)
            cv2.line(frame, self.floor_pts_2d[3], self.floor_pts_2d[0], (255, 255, 0), 1)

        # Draw Point 1 & 2 (Pink)
        if self.pos_point_1_2d is not None:
            cv2.circle(frame, self.pos_point_1_2d, 10, (180, 105, 255), -1) 
            cv2.circle(frame, self.pos_point_1_2d, 12, (255, 255, 255), 2)
            cv2.putText(frame, "P1", (self.pos_point_1_2d[0]+15, self.pos_point_1_2d[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 105, 255), 2)
        if self.pos_point_2_2d is not None:
            cv2.circle(frame, self.pos_point_2_2d, 10, (180, 105, 255), -1) 
            cv2.circle(frame, self.pos_point_2_2d, 12, (255, 255, 255), 2)
            cv2.putText(frame, "P2", (self.pos_point_2_2d[0]+15, self.pos_point_2_2d[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 105, 255), 2)
            if self.pos_point_1_2d is not None:
                cv2.line(frame, self.pos_point_1_2d, self.pos_point_2_2d, (180, 105, 255), 2)

        # Draw tracking circle (Red)
        if self.current_position_2d is not None and self.state in ["WARMUP", "TESTING", "FINISHED"]:
            cv2.circle(frame, self.current_position_2d, 10, (0, 0, 255), -1) 
            cv2.circle(frame, self.current_position_2d, 12, (255, 255, 255), 2)

    def run(self):
        self.start()
        
        while True:
            capture = self.kinect.get_capture()
            if capture.color is None or capture.transformed_depth is None:
                continue
            
            self.latest_depth = capture.transformed_depth
            color_img = capture.color[:, :, :3]
            color_rgb = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
            results = pose.process(color_rgb)
            self.current_position_2d = None 

            if results.pose_landmarks:
                l_ankle_lm = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE]
                r_ankle_lm = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE]

                l_ankle = self.get_3d_point(l_ankle_lm, color_img, capture.transformed_depth)
                r_ankle = self.get_3d_point(r_ankle_lm, color_img, capture.transformed_depth)
                
                if l_ankle is not None and r_ankle is not None:
                    self.current_position_3d = np.mean([l_ankle, r_ankle], axis=0)
                    self.l_ankle_height = self.calculate_height_to_floor(l_ankle)
                    self.r_ankle_height = self.calculate_height_to_floor(r_ankle)
                    
                    h, w, _ = color_img.shape
                    px_l, py_l = int(l_ankle_lm.x * w), int(l_ankle_lm.y * h)
                    px_r, py_r = int(r_ankle_lm.x * w), int(r_ankle_lm.y * h)
                    self.current_position_2d = (int((px_l + px_r) / 2), int((py_l + py_r) / 2))
                else:
                    self.current_position_3d = np.array([0.0, 0.0, 0.0])
                    self.l_ankle_height = 0.0
                    self.r_ankle_height = 0.0
                    self.current_position_2d = None

                # Handle Warmup Phase Transitions
                if self.state == "WARMUP":
                    elapsed = time.time() - self.start_time
                    if elapsed >= self.warmup_duration:
                        self.state = "TESTING"
                        self.start_time = time.time() # Reset timer for the actual test
                        self.step_count = 0
                        self.current_angle = 0.0
                        self.left_ankle_raised = False
                        self.right_ankle_raised = False
                        print("Warm-Up Complete. Recording started.")

                # Handle Testing Phase Logic
                elif self.state == "TESTING":
                    self.count_steps()
                    
                    if not np.array_equal(self.current_position_3d, np.array([0.0, 0.0, 0.0])):
                        self.current_angle = self.calculate_custom_angle(
                            self.pos_point_1_3d, 
                            self.pos_point_2_3d, 
                            self.current_position_3d
                        )
                    
                    if self.step_count >= self.target_steps:
                        self.state = "FINISHED"
                        self.duration = time.time() - self.start_time
                        self.final_position_3d = self.current_position_3d
                        self.final_angle = self.current_angle
                        print(f"Test Complete! Final Steps: {self.step_count}, Final Angle: {self.final_angle:.2f}")

            else:
                self.current_position_3d = np.array([0.0, 0.0, 0.0])
                self.l_ankle_height = 0.0
                self.r_ankle_height = 0.0
                self.current_position_2d = None
            
            # --- Rendering Phase ---
            if self.display_mode == "RGB":
                display_frame = color_img.copy()
            else:
                depth_vis = np.clip(capture.transformed_depth, 0, 4000)
                depth_vis = (depth_vis / 4000.0 * 255).astype(np.uint8)
                display_frame = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
                
            if results.pose_landmarks:
                red_spec = mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
                mp_drawing.draw_landmarks(
                    display_frame, 
                    results.pose_landmarks, 
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=red_spec,
                    connection_drawing_spec=red_spec
                )
            
            self.draw_ui(display_frame)
            cv2.imshow("FST System", display_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.kinect.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    system = FST_MediaPipe_System()
    system.run()