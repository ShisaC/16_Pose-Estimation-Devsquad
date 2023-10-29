from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import numpy as np
import mediapipe as mp
import cv2
import PoseModule as pm
import matplotlib.pyplot as plt
import base64

app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return render_template('login.html')


@app.route('/login', methods=['POST'])
def login():
    username = request.form['username']
    password = request.form['password']
    return render_template('home.html')

@app.route('/process_images', methods=['POST'])
def process_images():
    try:
        print(request)
        # Initialize MediaPipe Pose
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose()
        
        # Load images
        # Get the uploaded files from the request
        main_image = request.files['mainImage']
        comparison_image = request.files['comparisonImage']
        if main_image is None or comparison_image is None:
            return jsonify({'result': 'Please select between both main and comparison images.'})
        
        # Read and process the uploaded images
        main_image_data = main_image.read()
        comparison_image_data = comparison_image.read()

        main_image_np = np.frombuffer(main_image_data, np.uint8)
        comparison_image_np = np.frombuffer(comparison_image_data, np.uint8)

        main_image_rgb = cv2.imdecode(main_image_np, cv2.IMREAD_COLOR)
        comparison_image_rgb = cv2.imdecode(comparison_image_np, cv2.IMREAD_COLOR)
        detector = pm.PoseDetector()
        main_image_rgb = detector.findPose(main_image_rgb)
        main_landmarks = detector.getPosition(main_image_rgb)
        comparison_image_rgb = detector.findPose(comparison_image_rgb)
        comparison_landmarks = detector.getPosition(comparison_image_rgb)

        threshold = 150
        # Compare pose landmarks
        if main_landmarks and comparison_landmarks:
            similarity_score = calculate_similarity(main_landmarks, comparison_landmarks)
            if similarity_score < threshold:
                result = "Pose is correct"
            else:
                result = "Pose is wrong"
        else:
            result = "Pose data not available"
        main_image_rgb = draw_landmarks(main_image_rgb, main_landmarks)
        comparison_image_rgb = draw_landmarks(comparison_image_rgb, comparison_landmarks)

        return jsonify({"result": result,
                        "mainImage": image_to_base64(main_image_rgb),
                        "comparisonImage": image_to_base64(comparison_image_rgb)})
    except Exception as e:
        return jsonify({'result': f'Error: {str(e)}'}), 400

def calculate_similarity(main_landmarks, comparison_landmarks):
    main_points = np.array(main_landmarks)
    comparison_points = np.array(comparison_landmarks)
    # Calculate the Euclidean distances
    distances = np.linalg.norm(main_points[:, 1:] - comparison_points[:, 1:], axis=1)
    # Calculate the average distance
    average_distance = np.mean(distances)
    print(average_distance)
    return average_distance


def draw_landmarks(image, landmarks):
    print(landmarks)
    for landmark in landmarks:
        x, y, z = landmark
        cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
    return image


def image_to_base64(image):
    _, buffer = cv2.imencode(".jpg", image)
    image_base64 = base64.b64encode(buffer).decode()
    return image_base64


if __name__ == '__main__':
    app.run()
