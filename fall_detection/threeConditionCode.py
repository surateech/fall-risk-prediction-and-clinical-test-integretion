import os
import time
from typing import Dict, Tuple

import cv2
import numpy as np
import mediapipe as mp

from pyk4a import (
    PyK4A,
    Config,
    ColorResolution,
    DepthMode,
    FPS,
    depth_image_to_color_camera,
    CalibrationType,
)


# 1) PATH k4a.dll

os.environ["K4A_DLL_DIR"] = r"C:\Program Files\Azure Kinect SDK v1.4.2\sdk\windows-desktop\amd64\release\bin"


# 2) MEDIAPIPE CONFIG

mp_pose = mp.solutions.pose
LM = mp_pose.PoseLandmark

# index -> each joints
JOINT_LABELS = {
    0: "Head",
    1: "Shoulder center",
    2: "Shoulder right",
    3: "Elbow right",
    4: "Hand right",
    5: "Shoulder left",
    6: "Elbow left",
    7: "Hand left",
    8: "Hip right",
    9: "Knee right",
    10: "Ankle right",
    11: "Hip left",
    12: "Knee left",
    13: "Ankle left",
}

# skeletal
SIMPLE_BONES = [
    (1, 0),       # shoulder center - head
    (1, 2), (2, 3), (3, 4),      # ขวา: center -> shoulderR -> elbowR -> handR
    (1, 5), (5, 6), (6, 7),      # ซ้าย: center -> shoulderL -> elbowL -> handL
    (1, 8), (8, 9), (9, 10),     # ขา R
    (1, 11), (11, 12), (12, 13), # ขา L
]

# =========================
# 3) parameters for 3 conditions
# =========================
HIP_VEL_THRESH = 0.09       # m/s  (critical speed)
ANGLE_THRESH_DEG = 45.0     # body inclind
RATIO_THRESH = 1.0          # P = width / height


def create_kinect() -> PyK4A:
    """
    Open Azure Kinect with color + depth features and sync each frame
    """
    config = Config(
        color_resolution=ColorResolution.RES_720P,
        depth_mode=DepthMode.WFOV_2X2BINNED,
        camera_fps=FPS.FPS_30,
        synchronized_images_only=True,
    )
    k4a = PyK4A(config=config)
    k4a.start()
    print("Azure Kinect started.")
    return k4a


def compute_simple_joints_2d(
    pose_landmarks,
    W: int,
    H: int,
    vis_thresh: float = 0.5
) -> Dict[int, Tuple[int, int]]:
    """
    reduce mediapipe points from  33 to 14 
    give dict[index] = (u, v) in (pixel)
    """

    def get_uv(idx: int):
        lm = pose_landmarks[idx]
        if lm.visibility < vis_thresh:
            return None
        u = int(lm.x * W)
        v = int(lm.y * H)
        if u < 0 or u >= W or v < 0 or v >= H:
            return None
        return (u, v)

    pts: Dict[int, Tuple[int, int]] = {}

    # 0 — Head (NOSE)
    head = get_uv(LM.NOSE.value)
    if head:
        pts[0] = head

    # Left/Right shoulder
    l_sh = get_uv(LM.LEFT_SHOULDER.value)
    r_sh = get_uv(LM.RIGHT_SHOULDER.value)

    # 1 — Shoulder center = avg from l_sh and r_sh
    if l_sh and r_sh:
        pts[1] = ((l_sh[0] + r_sh[0]) // 2, (l_sh[1] + r_sh[1]) // 2)

    # 2 — Shoulder right
    if r_sh:
        pts[2] = r_sh

    # 5 — Shoulder left
    if l_sh:
        pts[5] = l_sh

    # 3 — Elbow right
    r_elb = get_uv(LM.RIGHT_ELBOW.value)
    if r_elb:
        pts[3] = r_elb

    # 4 — Hand right
    r_hand = get_uv(LM.RIGHT_WRIST.value)
    if r_hand:
        pts[4] = r_hand

    # 6 — Elbow left
    l_elb = get_uv(LM.LEFT_ELBOW.value)
    if l_elb:
        pts[6] = l_elb

    # 7 — Hand left
    l_hand = get_uv(LM.LEFT_WRIST.value)
    if l_hand:
        pts[7] = l_hand

    # 8,9,10 — Hip/Knee/Ankle right
    r_hip = get_uv(LM.RIGHT_HIP.value)
    r_knee = get_uv(LM.RIGHT_KNEE.value)
    r_ankle = get_uv(LM.RIGHT_ANKLE.value)
    if r_hip:
        pts[8] = r_hip
    if r_knee:
        pts[9] = r_knee
    if r_ankle:
        pts[10] = r_ankle

    # 11,12,13 — Hip/Knee/Ankle left
    l_hip = get_uv(LM.LEFT_HIP.value)
    l_knee = get_uv(LM.LEFT_KNEE.value)
    l_ankle = get_uv(LM.LEFT_ANKLE.value)
    if l_hip:
        pts[11] = l_hip
    if l_knee:
        pts[12] = l_knee
    if l_ankle:
        pts[13] = l_ankle

    return pts


def main():
    k4a = create_kinect()
    calib = k4a.calibration  # wrapper from k4a_calibration_t

    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    prev_joints_3d: Dict[int, Dict[str, float]] = {}
    prev_time: float = None

    # สำหรับ Condition 1: เก็บ y ของศูนย์กลางสะโพก (หน่วย m ใน camera frame)
    hip_prev_y_m: float = None
    hip_prev_time: float = None

    # state สำหรับ debug ว่าตรวจเจอ fall หรือยัง
    fall_state = "NORMAL"
    last_fall_time = None

    try:
        while True:
            capture = k4a.get_capture()
            color = capture.color
            depth = capture.depth

            if color is None or depth is None:
                continue

            # COLOR: BGRA -> BGR
            frame_bgra = color
            frame_bgr = frame_bgra[:, :, :3]
            frame_bgr = np.ascontiguousarray(frame_bgr, dtype=np.uint8)

            # DEPTH: align -> color camera geometry
            depth_in_color = depth_image_to_color_camera(
                depth,
                calib,
                thread_safe=True,
            )
            Hc, Wc = depth_in_color.shape[:2]

            # resize สีให้เท่ากับ depth
            frame_bgr_aligned = cv2.resize(frame_bgr, (Wc, Hc))
            frame_bgr_aligned = np.ascontiguousarray(frame_bgr_aligned, dtype=np.uint8)

            # RUN MEDIAPIPE POSE บน RGB
            frame_rgb = cv2.cvtColor(frame_bgr_aligned, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)

            # depth map 
            max_range_mm = 6000
            depth_vis = np.clip(depth_in_color, 0, max_range_mm).astype(np.float32)
            depth_vis = (depth_vis / max_range_mm * 255.0).astype(np.uint8)
            depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
            depth_vis = np.ascontiguousarray(depth_vis, dtype=np.uint8)

            now = time.time()

            # default vir of 3 conditions 
            hip_speed = None
            theta_deg = None
            rect_ratio = None
            M1 = 0
            M2 = 0
            M3 = 0

            if results.pose_landmarks:
                # 14 joint ในพิกัด depth/color (pixel)
                simple_pts = compute_simple_joints_2d(
                    results.pose_landmarks.landmark,
                    W=Wc,
                    H=Hc,
                    vis_thresh=0.5,
                )

                # present joint positions (u,v,X,Y,Z)
                curr_joints_3d: Dict[int, Dict[str, float]] = {}

                # 1) depth + calibration turn data to X,Y,Z (meter)
                for idx, (x, y) in simple_pts.items():
                    # sample depth around joint (3x3)
                    y0, y1 = max(y - 1, 0), min(y + 2, Hc)
                    x0, x1 = max(x - 1, 0), min(x + 2, Wc)
                    patch = depth_in_color[y0:y1, x0:x1]
                    valid = patch[patch > 0]
                    if valid.size == 0:
                        continue

                    z_mm = float(np.median(valid))

                    # (u,v,depth_mm) -> (X,Y,Z) mm ใน COLOR camera frame
                    xyz_mm = calib.convert_2d_to_3d(
                        (float(x), float(y)),
                        z_mm,
                        CalibrationType.COLOR,
                        CalibrationType.COLOR,
                    )

                    X_m = xyz_mm[0] / 1000.0
                    Y_m = xyz_mm[1] / 1000.0
                    Z_m = xyz_mm[2] / 1000.0

                    curr_joints_3d[idx] = {
                        "u": float(x),
                        "v": float(y),
                        "X": X_m,
                        "Y": Y_m,
                        "Z": Z_m,
                    }

                # 2) find v
                velocities: Dict[int, Dict[str, float]] = {}
                if prev_joints_3d and prev_time is not None:
                    dt = now - prev_time
                    if dt > 0:
                        for idx, curr in curr_joints_3d.items():
                            if idx not in prev_joints_3d:
                                continue
                            prev = prev_joints_3d[idx]
                            vx = (curr["X"] - prev["X"]) / dt
                            vy = (curr["Y"] - prev["Y"]) / dt
                            vz = (curr["Z"] - prev["Z"]) / dt
                            speed = float(np.sqrt(vx**2 + vy**2 + vz**2))
                            velocities[idx] = {
                                "vx": vx,
                                "vy": vy,
                                "vz": vz,
                                "speed": speed,
                            }

                # ===============================
                # 3) Calculate 3 condition
                # ===============================

                # ---- Condition 1 ----
                if 8 in curr_joints_3d and 11 in curr_joints_3d:
                    y8 = curr_joints_3d[8]["Y"]
                    y11 = curr_joints_3d[11]["Y"]
                    hip_center_y_m = 0.5 * (y8 + y11)  

                    if hip_prev_y_m is not None and hip_prev_time is not None:
                        dt_hip = now - hip_prev_time
                        if dt_hip > 0:
                            hip_speed = (hip_center_y_m - hip_prev_y_m) / dt_hip

                            if hip_speed >= HIP_VEL_THRESH:
                                M1 = 1
                    # update history
                    hip_prev_y_m = hip_center_y_m
                    hip_prev_time = now

                # ---- Condition 2 ----
                # ใช้ head (0) และ mid-ankle (10,13)
                if 0 in simple_pts and 10 in simple_pts and 13 in simple_pts:
                    x0, y0 = simple_pts[0]
                    x10, y10 = simple_pts[10]
                    x13, y13 = simple_pts[13]
                    xs = 0.5 * (x10 + x13)
                    ys = 0.5 * (y10 + y13)

                    dx = float(x0 - xs)
                    dy = float(y0 - ys)
                    # angle
                    theta_rad = np.arctan2(abs(dy), abs(dx) + 1e-6)
                    theta_deg = float(np.degrees(theta_rad))

                    if theta_deg < ANGLE_THRESH_DEG:
                        M2 = 1

                # ---- Condition 3: อัตราส่วน width / height ของ bounding box ----
                if len(simple_pts) >= 2:
                    xs_all = [p[0] for p in simple_pts.values()]
                    ys_all = [p[1] for p in simple_pts.values()]
                    w = float(max(xs_all) - min(xs_all))
                    h = float(max(ys_all) - min(ys_all))
                    if h > 0:
                        rect_ratio = w / h
                        if rect_ratio >= RATIO_THRESH:
                            M3 = 1

                # ----  3 conditions Conclusion ----
                fall_detected = (M1 == 1 and M2 == 1 and M3 == 1)

                if fall_detected:
                    fall_state = "FALL"
                    last_fall_time = now
                else:

                    #  RECOVERED
                    if fall_state == "FALL" and theta_deg is not None and rect_ratio is not None:
                        if theta_deg > ANGLE_THRESH_DEG and rect_ratio < RATIO_THRESH:
                            fall_state = "RECOVERED"

                # 4) visualize skeletal tracking
                for i, j in SIMPLE_BONES:
                    if i in curr_joints_3d and j in curr_joints_3d:
                        xi = int(curr_joints_3d[i]["u"])
                        yi = int(curr_joints_3d[i]["v"])
                        xj = int(curr_joints_3d[j]["u"])
                        yj = int(curr_joints_3d[j]["v"])
                        cv2.line(
                            depth_vis,
                            (xi, yi),
                            (xj, yj),
                            (0, 0, 255),  # แดง
                            2,
                        )

 
                for idx, pos in curr_joints_3d.items():
                    x = int(pos["u"])
                    y = int(pos["v"])
                    Z_m = pos["Z"]

                    cv2.circle(depth_vis, (x, y), 5, (255, 255, 255), -1)

                    label = JOINT_LABELS.get(idx, str(idx))
                    if idx in velocities:
                        sp = velocities[idx]["speed"]
                        text = f"{label}: {Z_m:.2f}m  v={sp:.2f} m/s"
                    else:
                        text = f"{label}: {Z_m:.2f}m"

                    cv2.putText(
                        depth_vis,
                        text,
                        (x + 5, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (255, 255, 255),
                        1,
                    )

                # debug
                if velocities:
                    print("---- 3D velocities (m/s) ----")
                    for idx in sorted(velocities.keys()):
                        name = JOINT_LABELS.get(idx, str(idx))
                        v = velocities[idx]
                        print(
                            f"{idx:2d} {name:16s}: "
                            f"vx={v['vx']:.2f}  vy={v['vy']:.2f}  "
                            f"vz={v['vz']:.2f}  | speed={v['speed']:.2f}"
                        )

                # update prev
                prev_joints_3d = curr_joints_3d
                prev_time = now

            # =========================================
            # 5)  3 conditions  (overlay)
            # =========================================
            overlay_y = 20
            line_dy = 22
            color_text = (255, 255, 255)

            # Condition 1
            if hip_speed is not None:
                txt = f"C1 Hip descent v={hip_speed:.3f} m/s  M1={M1}"
            else:
                txt = f"C1 Hip descent v=N/A  M1={M1}"
            cv2.putText(
                depth_vis,
                txt,
                (10, overlay_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color_text,
                2,
            )
            overlay_y += line_dy

            # Condition 2
            if theta_deg is not None:
                txt = f"C2 Body angle={theta_deg:.1f} deg  M2={M2}"
            else:
                txt = f"C2 Body angle=N/A  M2={M2}"
            cv2.putText(
                depth_vis,
                txt,
                (10, overlay_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color_text,
                2,
            )
            overlay_y += line_dy

            # Condition 3
            if rect_ratio is not None:
                txt = f"C3 Width/Height={rect_ratio:.2f}  M3={M3}"
            else:
                txt = f"C3 Width/Height=N/A  M3={M3}"
            cv2.putText(
                depth_vis,
                txt,
                (10, overlay_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color_text,
                2,
            )
            overlay_y += line_dy


            cv2.putText(
                depth_vis,
                f"State: {fall_state}",
                (10, overlay_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )


            if fall_state == "FALL":
                cv2.putText(
                    depth_vis,
                    "FALL DETECTED!",
                    (40, Hc // 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.5,
                    (0, 0, 255),
                    4,
                )

            cv2.imshow("Depth + 14-joint Skeleton + fall conditions", depth_vis)

            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord("q"):
                break

    finally:
        pose.close()
        k4a.stop()
        cv2.destroyAllWindows()
        print("Stopped and cleaned up.")


if __name__ == "__main__":
    main()
