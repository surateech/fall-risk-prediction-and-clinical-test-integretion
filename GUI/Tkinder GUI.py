import tkinter as tk
from tkinter import messagebox, ttk
import cv2
import mediapipe as mp
import numpy as np
import pyk4a
from pyk4a import Config, PyK4A, CalibrationType, ColorResolution, DepthMode, FPS
import time
import math
from collections import deque
from PIL import Image, ImageTk

# --- Helper Functions ---
def assess_stepping_risk(kmd, ratio, mhk):
    score = 0
    TH_KMD, TH_RATIO, TH_MHK = 5.0, 0.155, 0.080
    if kmd < TH_KMD: score += 1
    if ratio > TH_RATIO: score += 1
    if mhk < TH_MHK: score += 1
    return "High Fall" if score >= 2 else "Low Fall"

class FallGuardApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Fall Guard System v3.0 Professional")
        self.root.geometry("1400x900")
        self.root.configure(bg="#F4F7F9")
        
        # --- App Variables ---
        self.is_running = False
        self.current_mode = None 
        self.graph_data = deque(maxlen=200)
        
        # --- MediaPipe Setup ---
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.mp_drawing = mp.solutions.drawing_utils
        
        # --- Kinect Setup ---
        self.config = Config(
            color_resolution=ColorResolution.RES_720P,
            depth_mode=DepthMode.NFOV_UNBINNED,
            camera_fps=FPS.FPS_30,
            synchronized_images_only=True
        )
        try:
            self.k4a = PyK4A(config=self.config)
            self.k4a.start()
            self.calibration = self.k4a.calibration
        except Exception as e:
            messagebox.showerror("Camera Error", f"Cannot start Azure Kinect: {e}")
            self.root.destroy()
            return
            
        self.setup_ui()
        self.reset_all_variables()
        
        # Start main loop
        self.is_running = True
        self.update_frame()

    def setup_ui(self):
        # Header
        header = tk.Frame(self.root, bg="#1A237E", height=70)
        header.pack(fill="x")
        tk.Label(header, text="FALL GUARD SYSTEM: MULTI-FUNCTIONAL ASSESSMENT", 
                 fg="white", bg="#1A237E", font=("Helvetica", 18, "bold")).pack(pady=15)

        main_frame = tk.Frame(self.root, bg="#F4F7F9")
        main_frame.pack(expand=True, fill="both", padx=20, pady=20)

        # Left Column
        left_panel = tk.Frame(main_frame, bg="#F4F7F9", width=350)
        left_panel.pack(side="left", fill="y", padx=(0, 20))
        left_panel.pack_propagate(False)

        # Mode Selection
        mode_card = tk.LabelFrame(left_panel, text="Assessments", font=("Helvetica", 11, "bold"), bg="white", padx=15, pady=15)
        mode_card.pack(fill="x", pady=(0, 20))

        self.btn_rt = tk.Button(mode_card, text="1. Real-time Prediction", bg="#2962FF", fg="white", 
                                font=("Helvetica", 11), pady=10, command=lambda: self.change_mode("REALTIME"))
        self.btn_rt.pack(fill="x", pady=5)
        
        self.btn_tug = tk.Button(mode_card, text="2. TUG Test", bg="#5C6BC0", fg="white", 
                                 font=("Helvetica", 11), pady=10, command=lambda: self.change_mode("TUG"))
        self.btn_tug.pack(fill="x", pady=5)
        
        self.btn_step = tk.Button(mode_card, text="3. Stepping Test", bg="#26A69A", fg="white", 
                                  font=("Helvetica", 11), pady=10, command=lambda: self.change_mode("STEPPING"))
        self.btn_step.pack(fill="x", pady=5)

        ttk.Separator(mode_card, orient='horizontal').pack(fill='x', pady=10)

        self.btn_stop = tk.Button(mode_card, text="STOP / RESET", bg="#E53935", fg="white", 
                                  font=("Helvetica", 9, "bold"), pady=5, command=self.reset_all_variables)
        self.btn_stop.pack(fill="x", pady=5)

        # Dashboard Variables
        self.dash_vars = {
            "title": tk.StringVar(value="Select a mode to begin"),
            "rt_aspect": tk.StringVar(value="Aspect Ratio: -"),
            "rt_vel": tk.StringVar(value="Downward Velocity: -"),
            "rt_angle": tk.StringVar(value="Angle (deg): -"),
            "rt_risk": tk.StringVar(value="Fall Risk: -"),
            
            "tug_state": tk.StringVar(value="Status: -"),
            "tug_time": tk.StringVar(value="Time: -"),
            "tug_risk": tk.StringVar(value="Result: -"),
            
            "step_status": tk.StringVar(value="Status: -"),
            "step_th_kmd": tk.StringVar(value="TH_KMD: -"),
            "step_th_ratio": tk.StringVar(value="TH_RATIO: -"),
            "step_th_mhk": tk.StringVar(value="TH_MHK: -"),
            "step_result": tk.StringVar(value="Result: -")
        }

        # Dashboard UI
        self.dash_card = tk.LabelFrame(left_panel, text="Live Gait Dashboard", font=("Helvetica", 11, "bold"), bg="white", padx=15, pady=15)
        self.dash_card.pack(expand=True, fill="both")
        
        tk.Label(self.dash_card, textvariable=self.dash_vars["title"], font=("Helvetica", 12, "bold"), bg="white", fg="#333").pack(pady=(0,10))
        
        # Frames for dynamic display
        self.frame_rt = tk.Frame(self.dash_card, bg="white")
        tk.Label(self.frame_rt, textvariable=self.dash_vars["rt_aspect"], font=("Helvetica", 11), bg="white").pack(anchor="w", pady=2)
        tk.Label(self.frame_rt, textvariable=self.dash_vars["rt_vel"], font=("Helvetica", 11), bg="white").pack(anchor="w", pady=2)
        tk.Label(self.frame_rt, textvariable=self.dash_vars["rt_angle"], font=("Helvetica", 11), bg="white").pack(anchor="w", pady=2)
        tk.Label(self.frame_rt, textvariable=self.dash_vars["rt_risk"], font=("Helvetica", 14, "bold"), fg="red", bg="white").pack(anchor="w", pady=10)
        
        self.frame_tug = tk.Frame(self.dash_card, bg="white")
        tk.Label(self.frame_tug, textvariable=self.dash_vars["tug_state"], font=("Helvetica", 11), bg="white").pack(anchor="w", pady=2)
        tk.Label(self.frame_tug, textvariable=self.dash_vars["tug_time"], font=("Helvetica", 14), bg="white").pack(anchor="w", pady=2)
        tk.Label(self.frame_tug, textvariable=self.dash_vars["tug_risk"], font=("Helvetica", 14, "bold"), fg="red", bg="white").pack(anchor="w", pady=10)

        self.frame_step = tk.Frame(self.dash_card, bg="white")
        tk.Label(self.frame_step, textvariable=self.dash_vars["step_status"], font=("Helvetica", 12, "bold"), bg="white").pack(anchor="w", pady=2)
        tk.Label(self.frame_step, textvariable=self.dash_vars["step_th_kmd"], font=("Helvetica", 11), bg="white").pack(anchor="w", pady=2)
        tk.Label(self.frame_step, textvariable=self.dash_vars["step_th_ratio"], font=("Helvetica", 11), bg="white").pack(anchor="w", pady=2)
        tk.Label(self.frame_step, textvariable=self.dash_vars["step_th_mhk"], font=("Helvetica", 11), bg="white").pack(anchor="w", pady=2)
        tk.Label(self.frame_step, textvariable=self.dash_vars["step_result"], font=("Helvetica", 14, "bold"), fg="red", bg="white").pack(anchor="w", pady=10)

        # Right Column
        right_panel = tk.Frame(main_frame, bg="#F4F7F9")
        right_panel.pack(side="right", expand=True, fill="both")

        self.canvas_view = tk.Canvas(right_panel, bg="black", height=500)
        self.canvas_view.pack(fill="x", pady=(0, 10))
        self.canvas_view.bind("<Button-1>", self.mouse_callback)

        graph_frame = tk.LabelFrame(right_panel, text="Real-time Graph (Fall Risk %)", font=("Helvetica", 10), bg="white")
        graph_frame.pack(fill="both", expand=True)
        self.canvas_graph = tk.Canvas(graph_frame, bg="#111", height=200)
        self.canvas_graph.pack(fill="both", expand=True, padx=5, pady=5)

    def change_mode(self, mode):
        self.reset_all_variables()
        self.current_mode = mode
        
        self.frame_rt.pack_forget()
        self.frame_tug.pack_forget()
        self.frame_step.pack_forget()
        self.canvas_graph.delete("all")
        
        if mode == "REALTIME":
            self.dash_vars["title"].set("Real-time Prediction")
            self.frame_rt.pack(fill="both", expand=True)
            self.dash_vars["rt_aspect"].set("Aspect Ratio: 0.0")
            self.dash_vars["rt_vel"].set("Downward Velocity: 0.0")
            self.dash_vars["rt_angle"].set("Angle (deg): 0.0")
            self.dash_vars["rt_risk"].set("Fall Risk: 0.0%")
            messagebox.showinfo("Instructions", "Click 4 points on the floor in the camera view to establish the floor plane.")
            
        elif mode == "TUG":
            self.dash_vars["title"].set("TUG Test")
            self.frame_tug.pack(fill="both", expand=True)
            self.dash_vars["tug_state"].set("Status: WAITING")
            self.dash_vars["tug_time"].set("Time: 0.00 s")
            self.dash_vars["tug_risk"].set("Result: Pending...")
            self.tug_state = "WAITING"
            
        elif mode == "STEPPING":
            self.dash_vars["title"].set("Stepping Test")
            self.frame_step.pack(fill="both", expand=True)
            self.dash_vars["step_status"].set("Status: WAITING FOR USER")
            self.dash_vars["step_th_kmd"].set("")
            self.dash_vars["step_th_ratio"].set("")
            self.dash_vars["step_th_mhk"].set("")
            self.dash_vars["step_result"].set("")
            self.step_start_time = None

    def reset_all_variables(self):
        self.graph_data.clear()
        self.prev_time = time.time()
        
        # Realtime Variables
        self.floor_clicks = []
        self.floor_normal = None
        self.prev_com_y = None
        
        # TUG Variables
        self.tug_state = "WAITING"
        self.tug_start_time = None
        self.prev_com_3d_tug = None
        
        # Stepping Variables
        self.step_start_time = None
        self.is_recording_step = False
        self.step_finished = False
        self.max_movement_dist = 0.0
        self.TMD_sum = 0.0
        self.K_sum_L = 0.0
        self.K_sum_R = 0.0
        self.prev_K_L = None
        self.prev_K_R = None
        self.head_pos_last = None
        self.step_height_diffs_L = []
        self.step_height_diffs_R = []

    def get_3d_pos(self, landmark, width, height, depth_map):
        cx, cy = int(landmark.x * width), int(landmark.y * height)
        cx, cy = max(0, min(cx, width-1)), max(0, min(cy, height-1))
        
        roi = depth_map[max(0, cy-1):cy+2, max(0, cx-1):cx+2]
        valid_depths = roi[roi > 0]
        if len(valid_depths) == 0: return None
        
        depth_val = np.median(valid_depths)
        try:
            point_3d = self.calibration.convert_2d_to_3d((cx, cy), depth_val, CalibrationType.COLOR)
            return np.array(point_3d) / 1000.0 # Meters
        except: return None

    def mouse_callback(self, event):
        if self.current_mode == "REALTIME" and len(self.floor_clicks) < 4:
            # 1. ดึงขนาดความกว้างและความสูงของ Canvas ณ ปัจจุบัน
            canvas_w = self.canvas_view.winfo_width()
            canvas_h = self.canvas_view.winfo_height()
            
            # ป้องกันการหารด้วยศูนย์ในกรณีที่ Canvas ยังไม่แสดงผล
            if canvas_w > 0 and canvas_h > 0 and hasattr(self, 'orig_img_w'):
                
                # 2. แปลงพิกัดจาก Canvas กลับไปเป็นสเกลของภาพ OpenCV ต้นฉบับ (1280x720)
                real_x = int((event.x / canvas_w) * self.orig_img_w)
                real_y = int((event.y / canvas_h) * self.orig_img_h)
                
                # 3. บันทึกพิกัดจริงลงไป
                self.floor_clicks.append((real_x, real_y))
                
                if len(self.floor_clicks) == 4:
                    # Mock floor normal (สามารถเปลี่ยนกลับไปใช้ SVD คำนวณระนาบจริงได้หากต้องการ)
                    self.floor_normal = np.array([0, 1, 0]) 

    def draw_graph(self):
        # ลบข้อมูลเก่าทั้งหมดบน Canvas (รวมถึงเส้นและตัวหนังสือเก่า)
        self.canvas_graph.delete("all") 
        if len(self.graph_data) < 2: return
        
        w = self.canvas_graph.winfo_width()
        h = self.canvas_graph.winfo_height()
        
        # เว้นพื้นที่ด้านซ้ายไว้ 30 พิกเซลสำหรับเขียนตัวเลข
        left_margin = 30
        graph_w = w - left_margin
        
        # วาดตัวเลขบอกสเกล Y-Axis (0 - 100)
        self.canvas_graph.create_text(15, 10, text="100", fill="white", font=("Helvetica", 8))
        self.canvas_graph.create_text(15, h/2, text="50", fill="white", font=("Helvetica", 8))
        self.canvas_graph.create_text(15, h-10, text="0", fill="white", font=("Helvetica", 8))
        
        # วาดเส้น Grid ตรงกลาง (เกณฑ์ความเสี่ยง 50%)
        self.canvas_graph.create_line(left_margin, h/2, w, h/2, fill="gray", dash=(4, 4))
        
        # คำนวณจุดพิกัดของกราฟ
        pts = []
        x_step = graph_w / 200.0 # สมมติเก็บค่า 200 ค่า
        for i, val in enumerate(self.graph_data):
            px = left_margin + (i * x_step) # เริ่มวาดจากเส้นขอบ Margin
            py = h - (val / 100.0) * h      # คำนวณความสูงตามเปอร์เซ็นต์
            pts.append((px, py))
            
        # วาดเส้นกราฟเชื่อมแต่ละจุด
        for i in range(len(pts)-1):
            color = "red" if self.graph_data[i] > 50 else "#00FF00"
            self.canvas_graph.create_line(pts[i][0], pts[i][1], pts[i+1][0], pts[i+1][1], fill=color, width=2)

    def update_frame(self):
        if not self.is_running: return
        
        capture = self.k4a.get_capture()
        if capture.color is not None and capture.transformed_depth is not None:
            # 1. ดึงภาพ RGB สำหรับส่งให้ MediaPipe ประมวลผลเบื้องหลัง (ไม่แสดงผล)
            color_img = capture.color[:, :, :3].copy()
            h_img, w_img = color_img.shape[:2]
            self.orig_img_w = w_img
            self.orig_img_h = h_img
            depth_map = capture.transformed_depth
            
            # ประมวลผลโครงกระดูกด้วย MediaPipe
            results = self.pose.process(cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB))
            
            # 2. สร้างภาพ Depth Map สำหรับแสดงผลเพื่อความเป็นส่วนตัว
            depth_8bit = np.clip(depth_map / 4000.0 * 255.0, 0, 255).astype(np.uint8)
            display_img = cv2.applyColorMap(depth_8bit, cv2.COLORMAP_JET)
            
            # 3. วาดโครงกระดูกทับลงบนภาพ Depth (ไม่ใช่ภาพสี)
            if results.pose_landmarks:
                self.mp_drawing.draw_landmarks(display_img, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
            
            current_time = time.time()
            dt = current_time - self.prev_time
            self.prev_time = current_time

            # ----------------------------------------------------
            # 1. REALTIME PREDICTION LOGIC
            # ----------------------------------------------------
            if self.current_mode == "REALTIME":
                for pt in self.floor_clicks:
                    cv2.circle(display_img, pt, 5, (255, 0, 255), -1)
                    
                if results.pose_landmarks and self.floor_normal is not None:
                    lms = results.pose_landmarks.landmark
                    
                    # Aspect Ratio
                    xs = [lm.x * w_img for lm in lms]
                    ys = [lm.y * h_img for lm in lms]
                    dx, dy = max(xs) - min(xs), max(ys) - min(ys)
                    aspect_ratio = dx / dy if dy > 0 else 0.0
                    
                    # Velocity
                    hip_l = self.get_3d_pos(lms[23], w_img, h_img, depth_map)
                    hip_r = self.get_3d_pos(lms[24], w_img, h_img, depth_map)
                    downward_velocity = 0.0
                    if hip_l is not None and hip_r is not None:
                        com_y = (hip_l[1] + hip_r[1]) / 2.0
                        if self.prev_com_y is not None and dt > 0:
                            downward_velocity = (com_y - self.prev_com_y) / dt
                        self.prev_com_y = com_y

                    # Angle
                    nose_3d = self.get_3d_pos(lms[0], w_img, h_img, depth_map)
                    ankle_l_3d = self.get_3d_pos(lms[27], w_img, h_img, depth_map)
                    ankle_r_3d = self.get_3d_pos(lms[28], w_img, h_img, depth_map)
                    
                    angle_deg = 90.0 # ค่าเริ่มต้นหากกล้องจับจุดไม่เจอ
                    
                    if self.floor_normal is not None and nose_3d is not None and ankle_l_3d is not None and ankle_r_3d is not None:
                        ankle_mid_3d = (ankle_l_3d + ankle_r_3d) / 2.0
                        centerline_vec = ankle_mid_3d - nose_3d
                        mag_vec = np.linalg.norm(centerline_vec)
                        mag_norm = np.linalg.norm(self.floor_normal)
                        
                        if mag_vec > 0 and mag_norm > 0:
                            dot_prod = np.abs(np.dot(centerline_vec, self.floor_normal))
                            angle_rad = np.arcsin(dot_prod / (mag_vec * mag_norm))
                            angle_deg = np.degrees(angle_rad)

                    # Risk Calculation (Logistic Regression)
                    z = 0.0 + (5.0 * (downward_velocity - 0.09)) + (0.04 * (45 - angle_deg)) + (1.0 * (aspect_ratio - 1))
                    z_clamped = max(min(z, 50), -50)
                    fall_risk = (1.0 / (1.0 + math.exp(-z_clamped))) * 100.0
                    
                    self.graph_data.append(fall_risk)
                    
                    self.dash_vars["rt_aspect"].set(f"Aspect Ratio: {aspect_ratio:.2f}")
                    self.dash_vars["rt_vel"].set(f"Downward Velocity: {downward_velocity:.2f} m/s")
                    self.dash_vars["rt_angle"].set(f"Angle (deg): {angle_deg:.1f}")
                    self.dash_vars["rt_risk"].set(f"Fall Risk: {fall_risk:.1f}%")
                    self.draw_graph()

            # ----------------------------------------------------
            # 2. TUG TEST LOGIC
            # ----------------------------------------------------
            elif self.current_mode == "TUG":
                if results.pose_landmarks:
                    lms = results.pose_landmarks.landmark
                    hip_l, knee_l = lms[23].y * h_img, lms[25].y * h_img
                    hip_r, knee_r = lms[24].y * h_img, lms[26].y * h_img
                    
                    diff_l, diff_r = abs(hip_l - knee_l), abs(hip_r - knee_r)
                    posture = "Sitting" if (diff_l < h_img*0.1 and diff_r < h_img*0.1) else "Standing/Walking"
                    
                    if self.tug_state == "WAITING" and posture != "Sitting":
                        self.tug_start_time = time.time()
                        self.tug_state = "ACTIVE"
                        self.dash_vars["tug_state"].set("Status: ACTIVE")
                        
                    elif self.tug_state == "ACTIVE":
                        duration = time.time() - self.tug_start_time
                        self.dash_vars["tug_time"].set(f"Time: {duration:.2f} s")
                        
                        if posture == "Sitting" and duration > 2.0:
                            self.tug_state = "FINISHED"
                            self.dash_vars["tug_state"].set("Status: FINISHED")
                            risk = "LOW RISK" if duration < 13.5 else "HIGH RISK"
                            self.dash_vars["tug_risk"].set(f"Result: {risk}")

            # ----------------------------------------------------
            # 3. STEPPING TEST LOGIC
            # ----------------------------------------------------
            elif self.current_mode == "STEPPING":
                if not self.step_finished:
                    if self.step_start_time is None and results.pose_landmarks:
                        self.step_start_time = time.time()
                        
                    if self.step_start_time is not None:
                        elapsed = time.time() - self.step_start_time
                        
                        if elapsed < 10.0:
                            self.dash_vars["step_status"].set(f"WARM UP ({10.0 - elapsed:.1f}s)")
                        elif elapsed < 20.0:
                            if not self.is_recording_step:
                                self.is_recording_step = True
                                self.dash_vars["step_status"].set("RECORDING...")
                                self.dash_vars["step_th_kmd"].set("Current KMD: Calculating...")
                                self.dash_vars["step_th_ratio"].set("Current RATIO: Calculating...")
                                self.dash_vars["step_th_mhk"].set("Current MHK: Calculating...")
                                
                            if results.pose_landmarks:
                                lms = results.pose_landmarks.landmark
                                kL = self.get_3d_pos(lms[25], w_img, h_img, depth_map)
                                if kL is not None and self.prev_K_L is not None:
                                    self.K_sum_L += np.linalg.norm(kL - self.prev_K_L)
                                self.prev_K_L = kL
                                
                                self.dash_vars["step_th_kmd"].set(f"TH_KMD Track: {self.K_sum_L:.2f}")
                                
                        else:
                            self.step_finished = True
                            self.dash_vars["step_status"].set("FINISHED")
                            final_risk = assess_stepping_risk(self.K_sum_L, 0.1, 0.05)
                            self.dash_vars["step_result"].set(f"Result: {final_risk}")

            # 4. แสดงผลลง GUI (ใช้ภาพ Depth Map)
            display_rgb = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(display_rgb)
            
            # ปรับขนาดภาพให้พอดีกับ Canvas
            canvas_w = self.canvas_view.winfo_width()
            canvas_h = self.canvas_view.winfo_height()
            if canvas_w > 10 and canvas_h > 10:
                img_pil = img_pil.resize((canvas_w, canvas_h), Image.LANCZOS)
                
            self.img_tk = ImageTk.PhotoImage(img_pil)
            self.canvas_view.create_image(0, 0, anchor="nw", image=self.img_tk)

        self.root.after(30, self.update_frame)

    def on_closing(self):
        self.is_running = False
        if hasattr(self, 'k4a'):
            self.k4a.stop()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = FallGuardApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()