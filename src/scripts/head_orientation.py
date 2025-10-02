import cv2
import mediapipe as mp
import numpy as np
from utils.sound_functions import compute_directional_sound
import asyncio
import random
mp_face_mesh = mp.solutions.face_mesh

# Use 6+ reliable FaceMesh landmark indices
LANDMARKS = {
    "nose_tip": 1,
    "left_eye_outer": 33,
    "left_eye_inner": 133,
    "right_eye_inner": 362,
    "right_eye_outer": 263,
    "mouth_left": 61,
    "mouth_right": 291,
    "chin": 199
}

# Corresponding 3D model points (approximate, in mm). Must match order/length of image_points.
MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0),        # nose_tip
    (-30.0, -125.0, -30.0), # left_eye_outer
    (-10.0, -125.0, -30.0), # left_eye_inner
    (10.0, -125.0, -30.0),  # right_eye_inner
    (30.0, -125.0, -30.0),  # right_eye_outer
    (-60.0, -70.0, -60.0),  # mouth_left
    (60.0, -70.0, -60.0),   # mouth_right
    (0.0, -200.0, -5.0),    # chin
], dtype=np.float64)
user_pos = (0.0, 0.0)  # assume user at origin
angle = random.randint(-30,30)
dist = 0
target_pos = (dist * np.cos(np.radians(angle)), dist * np.sin(np.radians(angle)))
cap = cv2.VideoCapture(0)
with mp_face_mesh.FaceMesh(static_image_mode=False,
                           max_num_faces=1,
                           refine_landmarks=True,
                           min_detection_confidence=0.5,
                           min_tracking_confidence=0.5) as face_mesh:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0].landmark

            # build image_points array in the same order as MODEL_POINTS
            pts = []
            for k in ("nose_tip", "left_eye_outer", "left_eye_inner",
                      "right_eye_inner", "right_eye_outer",
                      "mouth_left", "mouth_right", "chin"):
                lm_idx = LANDMARKS[k]
                x = lm[lm_idx].x * w
                y = lm[lm_idx].y * h
                # skip frame if any landmark is invalid
                if np.isfinite(x) and np.isfinite(y):
                    pts.append((x, y))
                else:
                    pts = []
                    break

            if len(pts) != len(MODEL_POINTS):
                # insufficient/invalid landmarks this frame
                cv2.putText(frame, "Face landmarks unstable", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.imshow("Head Orientation", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
                continue

            image_points = np.array(pts, dtype="double")

            # Camera intrinsics (approx)
            focal_length = w
            center = (w / 2.0, h / 2.0)
            camera_matrix = np.array([
                [focal_length, 0, center[0]],
                [0, focal_length, center[1]],
                [0, 0, 1]
            ], dtype="double")
            dist_coeffs = np.zeros((4, 1))  # assume no lens distortion

            # Try a solver that works with 4+ points. Use ITERATIVE first, fallback to EPNP.
            success = False
            for flag in (cv2.SOLVEPNP_ITERATIVE, cv2.SOLVEPNP_EPNP):
                try:
                    res = cv2.solvePnP(MODEL_POINTS, image_points, camera_matrix, dist_coeffs, flags=flag)
                    # solvePnP returns (success, rvec, tvec) on newer OpenCV; on some builds it returns (rvec, tvec, success)
                    if isinstance(res, tuple) and len(res) == 3:
                        # (success, rvec, tvec)
                        possible_success, rot_vec, trans_vec = res
                        if isinstance(possible_success, (bool, np.bool_)):
                            success = bool(possible_success)
                        else:
                            # sometimes OpenCV returns rvec, tvec, success ordering
                            rot_vec, trans_vec, possible_success = res
                            success = bool(possible_success)
                    else:
                        # unexpected return; try unpacking two values (older)
                        rot_vec, trans_vec = res
                        success = True
                except cv2.error:
                    success = False

                if success:
                    break
            
            
            if not success:
                cv2.putText(frame, "solvePnP failed", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                rmat, _ = cv2.Rodrigues(rot_vec)
                proj_mat = np.hstack((rmat, trans_vec.reshape(3,1)))
                _, _, _, _, _, _, euler = cv2.decomposeProjectionMatrix(proj_mat)
                pitch, yaw, roll = [float(angle) for angle in euler]  # degrees
                user_heading_deg = yaw
                asyncio.run(compute_directional_sound(user_pos, user_heading_deg, target_pos))
                # Display orientation on the frame
                cv2.putText(frame, f"Yaw: {yaw:.1f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(frame, f"Pitch: {pitch:.1f}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(frame, f"Roll: {roll:.1f}", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("Head Orientation", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
