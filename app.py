from flask import Flask, render_template, request, Response, jsonify, redirect, url_for
import cv2
import numpy as np
import mediapipe as mp
import pandas as pd
import pickle

app = Flask(__name__)

exercise_state = {
    'bicep': {'counter': 0, 'current_stage': '', 'bodylang_prob': 0.0, 'timeElapsed': 0},
    'squat': {'counter': 0, 'form_stage': '', 'valid_rep': False, 'hip_knee_angle': 0, 'timeElapsed': 0},
    'deadlift': {'counter': 0, 'hip_knee_angle': 0, 'hand_status': 'OK', 'stage': '', 'timeElapsed': 0}
}

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/newstats')
def newstats():
    kcal_bicep = compute_kcal('bicep', exercise_state['bicep']['counter'], exercise_state['bicep']['timeElapsed'])
    kcal_squat = compute_kcal('squat', exercise_state['squat']['counter'], exercise_state['squat']['timeElapsed'])
    kcal_deadlift = compute_kcal('deadlift', exercise_state['deadlift']['counter'], exercise_state['deadlift']['timeElapsed'])
    total_kcal = round(kcal_bicep + kcal_squat + kcal_deadlift,2)

    return render_template('newstats.html', exercise_state=exercise_state,
                           kcal_bicep=kcal_bicep, kcal_squat=kcal_squat,
                           kcal_deadlift=kcal_deadlift, total_kcal=total_kcal)


@app.route('/bicep')
def bicep():
    return render_template('bicep.html')

@app.route('/save_time', methods=['POST'])
def save_time():
    data = request.get_json()
    exercise = data.get('exercise')
    time_elapsed = data.get('timeElapsed')

    if exercise in exercise_state:
        exercise_state[exercise]['timeElapsed'] = time_elapsed  # Save elapsed time
        return jsonify({'message': f'Time saved for {exercise}', 'timeElapsed': time_elapsed}), 200

    return jsonify({'error': 'Invalid exercise name'}), 400

@app.route('/squats')
def squats():
    return render_template('squats.html')

@app.route('/deadlift')
def deadlift():
    return render_template('deadlift.html')

@app.route('/video_feed/<exercise>')
def video_feed(exercise):
    if exercise == "bicep":
        return Response(generate_frames('bicep'), mimetype='multipart/x-mixed-replace; boundary=frame')
    elif exercise == "squat":
        return Response(generate_frames('squat'), mimetype='multipart/x-mixed-replace; boundary=frame')
    elif exercise == "deadlift":
        return Response(generate_frames('deadlift'), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/exercise_data/<exercise>')
def exercise_data(exercise):
    if exercise in exercise_state:
        print(f"Sending data for {exercise}: {exercise_state[exercise]}") #Debugging
        return jsonify(exercise_state[exercise])
    return jsonify({'error': 'Invalid exercise'}), 400

@app.route('/reset/<exercise>', methods=['POST'])
def reset(exercise):
    if exercise in exercise_state:
        for key in exercise_state[exercise]:
            if isinstance(exercise_state[exercise][key], int):
                exercise_state[exercise][key] = 0
            elif isinstance(exercise_state[exercise][key], str):
                exercise_state[exercise][key] = ''
            elif isinstance(exercise_state[exercise][key], bool):
                exercise_state[exercise][key] = False
    return jsonify({'message': f'{exercise.capitalize()} counter reset'})


# Helper function
def compute_kcal(exercise, counter, time_elapsed):
    mets = {"bicep": 4.5, "squat": 5.0, "deadlift": 6.0}
    met = mets.get(exercise, 4.5)
    print("time elapsed for", exercise, "is", time_elapsed)
    time_in_hours = time_elapsed / 3600
    intensity_factor = 1 + (counter / 1000)
    return round(met * time_in_hours * intensity_factor, 2)


def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360.0 - angle if angle > 180.0 else angle

def generate_frames(exercise):

    # Mediapipe initialization
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_tracking_confidence=0.7, min_detection_confidence=0.7)

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(106, 13, 173), thickness=4, circle_radius=5),
                mp_drawing.DrawingSpec(color=(255, 102, 0), thickness=5, circle_radius=10)
            )

            update_exercise_state(exercise, results.pose_landmarks.landmark)

        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    cap.release()
        
# Load trained Bicep Curl model
with open('bicep_curl.pkl', 'rb') as f:
    model = pickle.load(f)

# with open('deadlift.pkl', 'rb') as f:
#     deadlift_model = pickle.load(f)

def update_exercise_state(exercise, landmarks):

    if exercise == 'bicep' :
        # Flatten pose landmarks for prediction
        row = np.array([[res.x, res.y, res.z, res.visibility] for res in landmarks]).flatten().tolist()
        X = pd.DataFrame([row])

        # Model predictions
        bodylang_prob = model.predict_proba(X)[0]
        bodylang_class = model.predict(X)[0]

        # Update prediction in exercise state
        exercise_state['bicep']['bodylang_prob'] = round(bodylang_prob[bodylang_prob.argmax()], 2)

        # Movement logic
        current_stage = exercise_state['bicep']['current_stage']
        if bodylang_class == 1 and bodylang_prob[bodylang_prob.argmax()] > 0.7:
            exercise_state['bicep']['current_stage'] = "up"
        elif current_stage == "up" and bodylang_class == 0 and bodylang_prob[bodylang_prob.argmax()] > 0.7:
            exercise_state['bicep']['current_stage'] = "down"
            exercise_state['bicep']['counter'] += 1

    elif exercise == 'squat':
        # Calculate joint angles relevant to squat
        left_hip = [landmarks[23].x, landmarks[23].y]
        left_knee = [landmarks[25].x, landmarks[25].y]
        left_ankle = [landmarks[27].x, landmarks[27].y]
        hip_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)

        # Store the calculated angle in the dictionary
        exercise_state['squat']['hip_knee_angle'] = round(hip_knee_angle, 2)

        # Classify squat form
        if hip_knee_angle < 130:
            squat_form = "Deep Squat"
            exercise_state['squat']['valid_rep'] = False  # Invalidate the rep
        elif 130 <= hip_knee_angle <= 150:
            squat_form = "Good Squat"
            if exercise_state['squat'].get('current_stage') != "down":
                exercise_state['squat']['valid_rep'] = True  # Mark as a valid rep
            exercise_state['squat']['current_stage'] = "down"
        else:
            squat_form = "Shallow Squat"
            if exercise_state['squat'].get('current_stage') == "down" and exercise_state['squat']['valid_rep']:
                exercise_state['squat']['counter'] += 1  # Increment counter for a valid rep
                exercise_state['squat']['valid_rep'] = False  # Reset valid rep
            exercise_state['squat']['current_stage'] = "up"

        # Update the form stage in the dictionary
        exercise_state['squat']['form_stage'] = squat_form

    elif exercise == 'deadlift':
        left_hip = [landmarks[23].x, landmarks[23].y]
        left_knee = [landmarks[25].x, landmarks[25].y]
        left_ankle = [landmarks[27].x, landmarks[27].y]

        left_wrist = [landmarks[15].x, landmarks[15].y]
        right_wrist = [landmarks[16].x, landmarks[16].y]

        # Calculate the hip-knee angle
        hip_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
        exercise_state['deadlift']['hip_knee_angle'] = round(hip_knee_angle, 2)

        # Calculate hand position (horizontal distance)
        hand_distance = abs(left_wrist[0] - right_wrist[0])
        hand_status = "Wide" if hand_distance > 0.22 else "OK"
        exercise_state['deadlift']['hand_status'] = hand_status

        # Classify stage and increment counter
        if hip_knee_angle > 160:
            exercise_state['deadlift']['stage'] = "up"
        elif hip_knee_angle < 145 and exercise_state['deadlift']['stage'] == "up":
            exercise_state['deadlift']['stage'] = "down"
            exercise_state['deadlift']['counter'] += 1

if __name__ == '__main__':
    app.run(debug=True)