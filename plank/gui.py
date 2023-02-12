import threading
import time

import mediapipe as mp
import cv2
import numpy as np
import pickle
import pandas as pd
from plank import landmarks

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic


timer_count = 0
highest_count = 0
timer_started = False
thr = None

with open('plank.pkl', 'rb') as f:
    model = pickle.load(f)


def count_up():
    while timer_started is True:
        global timer_count
        timer_count += 1

        # if timer count is above highest count, set highest count:
        if timer_count >= highest_count:
            timer_count += 1

        # sleep timer with one second interval
        time.sleep(1)
        print(f"Timer: {timer_count}")

cap = cv2.VideoCapture(0)
# Initiate holistic model
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()

        # Recolor Feed
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Make Detections
        results = holistic.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # 4. Pose Detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                                  mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                  )
        # Export coordinates
        try:
            pose = results.pose_landmarks.landmark
            pose_row = list(
                np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())

            row = pose_row

            df = pd.DataFrame([row], columns=landmarks)
            body_language_class = model.predict(df)[0]
            body_language_prob = model.predict_proba(df)[0]
            # print(body_language_class, body_language_prob, str(round(body_language_prob[np.argmax(body_language_prob)], 2)))

            cv2.rectangle(image, (0, 0), (250, 60), (245, 117, 16), -1)


            if body_language_class.split(' ')[0] == "plank":
                if round(body_language_prob[np.argmax(body_language_prob)], 2) > 0.6:
                    if not timer_started:
                        print("Detected plank, starting timer")
                        timer_started = True
                        # create a timer in a separate thread
                        thr = threading.Thread(target=count_up)
                        thr.start()
                elif round(body_language_prob[np.argmax(body_language_prob)], 2) < 0.59:
                    if timer_started:
                        timer_started = False
                        timer_count = 0
                        if thr is not None:
                            thr.join()
                            thr = None
                        print("Detected not plank, stopping timer (PROB)")
            else:
                if timer_started:
                    timer_started = False
                    timer_count = 0
                    if thr is not None:
                        thr.join()
                        thr = None
                    print("Detected not plank, stopping timer")

            cv2.putText(image, 'TIMER'
                        , (95, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, str(timer_count)
                        , (90, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            # Display Probability
            cv2.putText(image, 'HIGHEST COUNT'
                        , (15, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, str(highest_count)
                        , (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        except:
            pass

        cv2.imshow('Raw Webcam Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

