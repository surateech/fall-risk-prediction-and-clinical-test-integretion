import cv2
import mediapipe as mp
import numpy as np
import pyk4a
from pyk4a import Config, PyK4A
from scipy.signal import butter, lfilter, lfilter_zi # เพิ่ม lfilter_zi
import time

# --- 1. Class กรองสัญญาณ (Butterworth Filter) แบบ Real-time ---
class LowPassFilter:
    def __init__(self, cutoff=6.0, fs=30.0, order=4):
        # [cite_start]ออกแบบ Filter ตามทฤษฎี [cite: 77]
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        self.b, self.a = butter(order, normal_cutoff, btype='low', analog=False)
        
        # ตัวแปรเก็บสถานะ (Memory) ของ Filter
        self.zi = None

    def process(self, data_point):
        """
        กรองข้อมูล 3 มิติ (x, y, z) ทีละเฟรม
        data_point: numpy array shape (3,)
        """
        # ถ้าเป็นเฟรมแรก ให้สร้างสถานะเริ่มต้น (Initial State)
        if self.zi is None:
            # สร้าง zi เริ่มต้นสำหรับ 3 channels (x, y, z)
            # lfilter_zi คืนค่า state เริ่มต้นของ filter 1 ตัว
            zi_per_channel = lfilter_zi(self.b, self.a)
            # ขยายให้รองรับ 3 แกน
            self.zi = np.array([zi_per_channel * val for val in data_point]).T

        # กรองข้อมูลทีละแกน (X, Y, Z)
        filtered_point = []
        for i in range(len(data_point)):
            # กรองค่าปัจจุบันโดยอ้างอิงสถานะเดิม (zi)
            out, new_zi = lfilter(self.b, self.a, [data_point[i]], zi=self.zi[:, i])
            # อัปเดตสถานะใหม่เก็บไว้ใช้รอบหน้า
            self.zi[:, i] = new_zi
            filtered_point.append(out[0])
            
        return np.array(filtered_point)

# --- 2. Class หลักสำหรับคำนวณ Stride Length ---
class StrideLengthCalculator:
    def __init__(self):
        # MediaPipe Setup
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # [จุดที่เพิ่ม 1] สร้างตัวกรองแยกสำหรับแต่ละจุดที่สำคัญ
        self.filter_sacrum = LowPassFilter()
        self.filter_heel_l = LowPassFilter()
        self.filter_heel_r = LowPassFilter()

        # ตัวแปรเก็บสถานะการเดิน
        self.prev_heel_pos = {'left': None, 'right': None} 
        self.stride_data = {
            'left': {'x': 0.0, 'z': 0.0, 'total': 0.0},
            'right': {'x': 0.0, 'z': 0.0, 'total': 0.0}
        }
        self.step_counts = {'left': 0, 'right': 0}
        
        # History สำหรับ Zeni Algorithm
        self.relative_pos_history = {'left': [], 'right': []}

    def get_landmark_3d(self, landmarks, idx, depth_map, calibration):
        # ฟังก์ชันแปลงพิกัด (เหมือนเดิม แต่รวม Fix CalibrationType.COLOR แล้ว)
        h, w = depth_map.shape
        lm = landmarks[idx]
        cx, cy = int(lm.x * w), int(lm.y * h)
        
        if cx < 0 or cx >= w or cy < 0 or cy >= h:
            return None

        z_region = depth_map[max(0, cy-1):min(h, cy+2), max(0, cx-1):min(w, cx+2)]
        if z_region.size == 0: return None
        depth_mm = np.median(z_region)
        
        if depth_mm == 0: return None

        # Fix: ใส่ CalibrationType.COLOR ตามที่แก้ไปก่อนหน้า
        point_3d = calibration.convert_2d_to_3d(
            (cx, cy), 
            depth_mm,
            pyk4a.CalibrationType.COLOR
        )
        
        return np.array(point_3d) / 1000.0 # เมตร

    def process_frame(self, color_image, depth_map, calibration):
        results = self.pose.process(cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB))
        current_data = {}
        
        if results.pose_landmarks:
            lms = results.pose_landmarks.landmark
            
            # 1. ดึงข้อมูล "ดิบ" (Raw Data) จาก MediaPipe + Kinect
            hip_l = self.get_landmark_3d(lms, 23, depth_map, calibration)
            hip_r = self.get_landmark_3d(lms, 24, depth_map, calibration)
            heel_l_raw = self.get_landmark_3d(lms, 29, depth_map, calibration)
            heel_r_raw = self.get_landmark_3d(lms, 30, depth_map, calibration)

            if hip_l is not None and hip_r is not None and heel_l_raw is not None and heel_r_raw is not None:
                sacrum_raw = (hip_l + hip_r) / 2.0
                
                # [จุดที่เพิ่ม 2] นำค่าดิบผ่าน Filter ก่อนนำไปใช้
                sacrum_pos = self.filter_sacrum.process(sacrum_raw)
                heel_l = self.filter_heel_l.process(heel_l_raw)
                heel_r = self.filter_heel_r.process(heel_r_raw)
                
                # ส่งค่าที่ "นิ่ง" แล้ว ไปคำนวณหา Stride
                self.detect_heel_strike('left', heel_l, sacrum_pos)
                self.detect_heel_strike('right', heel_r, sacrum_pos)
                
                # เก็บค่าเพื่อส่งไปวาดบนหน้าจอ
                current_data['heel_l'] = heel_l
                current_data['heel_r'] = heel_r

        return current_data, results

    def detect_heel_strike(self, side, heel_pos_3d, sacrum_pos_3d):
        # Algorithm ที่แก้ให้จับแกน X ได้ด้วย (ใช้ Euclidean Distance)
        dist_from_center = np.sqrt(
            (heel_pos_3d[0] - sacrum_pos_3d[0])**2 + 
            (heel_pos_3d[2] - sacrum_pos_3d[2])**2
        )
        
        history = self.relative_pos_history[side]
        history.append(dist_from_center)
        if len(history) > 5: history.pop(0)
        
        if len(history) >= 3:
            # Check Peak (จังหวะก้าวไกลสุด)
            if history[-2] > history[-1] and history[-2] > history[-3]:
                 
                 # Threshold 0.2m กัน Noise ตอนยืน
                 if history[-2] > 0.2:
                     
                     current_heel_pos = heel_pos_3d
                     
                     if self.prev_heel_pos[side] is None:
                         pass # ก้าวแรก เก็บค่าอย่างเดียว
                     else:
                         p1 = self.prev_heel_pos[side]
                         p2 = current_heel_pos
                         
                         # คำนวณระยะแยกแกน
                         dist_x = abs(p2[0] - p1[0])
                         dist_z = abs(p2[2] - p1[2])
                         dist_total = np.sqrt(dist_x**2 + dist_z**2)
                         
                         if 0.1 < dist_total < 2.5:
                            self.stride_data[side]['x'] = dist_x
                            self.stride_data[side]['z'] = dist_z
                            self.stride_data[side]['total'] = dist_total
                            self.step_counts[side] += 1
                     
                     self.prev_heel_pos[side] = current_heel_pos

# --- 3. Main Loop & UI ---
def main():
    config = Config(
        color_resolution=pyk4a.ColorResolution.RES_720P,
        depth_mode=pyk4a.DepthMode.NFOV_UNBINNED,
        camera_fps=pyk4a.FPS.FPS_30,
        synchronized_images_only=True,
    )
    
    k4a = PyK4A(config)
    k4a.start()

    stride_calc = StrideLengthCalculator()

    try:
        while True:
            capture = k4a.get_capture()
            if capture.color is None or capture.depth is None:
                continue

            transformed_depth = capture.transformed_depth
            color_img = capture.color[:, :, :3]
            
            data_3d, mp_results = stride_calc.process_frame(
                color_img, 
                transformed_depth, 
                k4a.calibration
            )

            # UI Visualization
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(transformed_depth, alpha=0.03), 
                cv2.COLORMAP_JET
            )
            
            if mp_results.pose_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(
                    depth_colormap,
                    mp_results.pose_landmarks,
                    mp.solutions.pose.POSE_CONNECTIONS
                )

            # Overlay Box
            cv2.rectangle(depth_colormap, (0, 0), (400, 380), (20, 20, 20), -1)
            cv2.addWeighted(depth_colormap, 1, depth_colormap, 0, 0, depth_colormap)
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            
            cv2.putText(depth_colormap, "STRIDE ANALYZER (FILTERED)", (15, 30), font, 0.7, (0, 255, 255), 2)
            
            # Display Left Leg
            y_start = 70
            step = 30
            l_data = stride_calc.stride_data['left']
            cv2.putText(depth_colormap, f"LEFT (Steps: {stride_calc.step_counts['left']})", (15, y_start), font, 0.6, (150, 255, 150), 2)
            cv2.putText(depth_colormap, f"Stride X : {l_data['x']:.3f} m", (30, y_start + step), font, 0.6, (200, 255, 200), 1)
            cv2.putText(depth_colormap, f"Stride Z : {l_data['z']:.3f} m", (30, y_start + step*2), font, 0.6, (200, 255, 200), 1)
            cv2.putText(depth_colormap, f"Total    : {l_data['total']:.3f} m", (30, y_start + step*3), font, 0.6, (0, 255, 0), 2)
            
            # Display Right Leg
            y_start_r = y_start + step*4 + 20
            r_data = stride_calc.stride_data['right']
            cv2.putText(depth_colormap, f"RIGHT (Steps: {stride_calc.step_counts['right']})", (15, y_start_r), font, 0.6, (150, 150, 255), 2)
            cv2.putText(depth_colormap, f"Stride X : {r_data['x']:.3f} m", (30, y_start_r + step), font, 0.6, (200, 200, 255), 1)
            cv2.putText(depth_colormap, f"Stride Z : {r_data['z']:.3f} m", (30, y_start_r + step*2), font, 0.6, (200, 200, 255), 1)
            cv2.putText(depth_colormap, f"Total    : {r_data['total']:.3f} m", (30, y_start_r + step*3), font, 0.6, (50, 100, 255), 2)

            cv2.imshow("Kinect Stride Analysis", depth_colormap)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        k4a.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()