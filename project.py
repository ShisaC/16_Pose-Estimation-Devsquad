
import cv2
import mediapipe as mp
import numpy as np
from datetime import datetime

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
        
    return angle

def is_squat(hip, knee, ankle):
    angle_knee = calculate_angle(hip, knee, ankle)
    return angle_knee <= 90  

def is_push_up(shoulder, elbow, hip, knee):
    angle_elbow = calculate_angle(shoulder, elbow, hip)
    angle_knee = calculate_angle(hip, knee, hip)
    
    return angle_elbow <= 90 and angle_knee <= 90  

def is_leg_raise(hip, knee, ankle,shoulder):
    angle_knee = calculate_angle(hip, knee, ankle)
    angle_hip = calculate_angle(shoulder, hip, knee)
    return 150 < angle_knee <= 210 and 60 < angle_hip <= 120  

def is_sit_up(shoulder, hip, knee):
    angle_hip = calculate_angle(shoulder, hip, knee)
    return 60 < angle_hip <= 120  

def is_tadasana(shoulder, hip, knee, ankle,wrist):
    angle_hip = calculate_angle(shoulder, hip, knee)
    angle_ankle = calculate_angle(hip, knee, ankle)
    angle_shoulder = calculate_angle(wrist, shoulder, hip)
    return 150 < angle_shoulder <= 210 and 150 < angle_hip <= 210 and 150 < angle_ankle <= 210  

def is_bridge(shoulder, hip, knee, ankle):
    angle_hip = calculate_angle(shoulder, hip, knee)
    angle_ankle = calculate_angle(hip, knee, ankle)
    return 150 <=angle_hip <= 230 and 50 <=angle_ankle <= 120  

def is_kneepush_up(shoulder, elbow, hip, knee,ankle):
    angle_elbow = calculate_angle(shoulder, elbow, hip)
    angle_knee = calculate_angle(hip, knee, ankle)
    
    return angle_elbow <= 90 and angle_knee <= 90  

def is_t_pose(shoulder, hip, elbow, ankle):
    angle_hip = calculate_angle(shoulder, hip, ankle)  
    angle_shoulder = calculate_angle(elbow, shoulder, hip)  
    
    return 160 < angle_hip < 200 and 80 < angle_shoulder < 100 


def process_webcam(exercise):
    cap = cv2.VideoCapture(0)

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        squat_stage = None
        push_up_stage = None
        leg_raise_stage = None
        sit_up_stage = None
        tadasana_stage = None
        t_pose_stage = None
        exercise_count = 0
        start_time = datetime.now()
        end_time = None
        frame_counter = 0
        frames_to_process = 30
        while cap.isOpened():
            
            ret, frame = cap.read()

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            results = pose.process(image)

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            try:
                landmarks = results.pose_landmarks.landmark
                hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
                shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

                for landmark in mp_pose.PoseLandmark:
                    cv2.circle(image, (int(landmarks[landmark.value].x * frame.shape[1]), int(landmarks[landmark.value].y * frame.shape[0])), 5, (0, 255, 0), -1)

                
                for connection in mp_pose.POSE_CONNECTIONS:
                    start_point = connection[0]
                    end_point = connection[1]

                    x1, y1 = int(landmarks[start_point].x * frame.shape[1]), int(landmarks[start_point].y * frame.shape[0])
                    x2, y2 = int(landmarks[end_point].x * frame.shape[1]), int(landmarks[end_point].y * frame.shape[0])

                    cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

                exercise_map = {
                    "squat": is_squat(hip, knee, ankle),
                    "leg_raise": is_leg_raise(hip, knee, ankle,shoulder),
                    "push_up": is_push_up(shoulder, elbow, hip, knee),
                    "sit_up": is_sit_up(shoulder, hip, knee),
                    "tadasana": is_tadasana(shoulder, hip, knee, ankle,wrist),
                    "glute_bridge": is_bridge(shoulder, hip, knee, ankle),
                    "knee_push_up": is_kneepush_up(shoulder, elbow, hip, knee,ankle),
                    "t_pose": is_t_pose(shoulder, hip, elbow, ankle),
                }
                if frame_counter == frames_to_process:
                    stage = exercise_map[exercise]
                    if stage:
                        exercise_count += 1
                    frame_counter = 0
            except:
                pass

            cv2.putText(image, f"{exercise} count: {exercise_count}", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.imshow('Pose Detection', image)

            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'): 
                end_time = datetime.now()       
                break
            frame_counter += 1

        time_difference = end_time - start_time
        hours, remainder = divmod(time_difference.seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        cap.release()
        cv2.destroyAllWindows()
        return {
            "exercise_count": exercise_count,
            "exercise": exercise,
            "duration": f"{hours:02}:{minutes:02}:{seconds:02}",
            "start_time": start_time.strftime("%Y-%m-%d %H:%M:%S")
        }


if __name__ == "__main__":
    process_webcam()

