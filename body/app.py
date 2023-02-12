import csv

import mediapipe as mp
import cv2
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

class_name = "right_head"


def orginal():
    cap = cv2.VideoCapture(0)
    with mp_holistic.Holistic(min_tracking_confidence=0.5, min_detection_confidence=0.5) as holistic:

        while cap.isOpened():
            ret, image = cap.read()

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = holistic.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # 1. Draw face landmarks
            mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
                                        mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
                                        mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1)
                                        )

            # 2. Right hand
            mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                        mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
                                        mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2)
                                        )

            # 3. Left Hand
            mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                        mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                                        mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
                                        )

            # 4. Pose Detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                        mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                                        mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                        )

            cv2.imshow("MediaPipe Pose", image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break


def write_coords():
    """
    Write the coordinates of the landmarks to a csv file
    :return: None
    """
    num_coords = len(results.pose_landmarks.landmark)+len(results.face_landmarks.landmark)

    landmarks = ["class"]
    for val in range(1, num_coords+1):
        landmarks += [f"x{val}", f"y{val}", f"z{val}", f"v{val}"]

    print(len(landmarks))

    # with open('coords.csv', mode='w') as f:
    #    csv_writer = csv.writer(f)
    #    csv_writer.writerow(landmarks)

cap = cv2.VideoCapture(0)
with mp_holistic.Holistic(min_tracking_confidence=0.5, min_detection_confidence=0.5) as holistic:

    while cap.isOpened():
        ret, image = cap.read()

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = holistic.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # 1. Draw face landmarks
        mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
                                  mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
                                  mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1)
                                  )

        # 2. Right hand
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
                                  mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2)
                                  )

        # 3. Left Hand
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                                  mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
                                  )

        # 4. Pose Detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                                  mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                  )

        try:
            pose = results.pose_landmarks.landmark
            pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())

            face = results.face_landmarks.landmark
            face_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in face]).flatten())

            row = pose_row + face_row
            row.insert(0, class_name)

            #with open("coords.csv", mode="a", newline="") as f:
            #   csv_writer = csv.writer(f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL)
            #   csv_writer.writerow(row)

        except:
            pass

        cv2.imshow("MediaPipe Pose", image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

write_coords()

def write_landmarks():
    """
    Write the coordinates of the landmarks to a csv file
    :return: None
    """
    num_coords = len(results.pose_landmarks.landmark)+len(results.face_landmarks.landmark)

    landmarks = ["class"]
    for val in range(1, num_coords+1):
        landmarks += [f'x{val}', f"y{val}", f"z{val}", f"v{val}"]

    with open('landmarks.py', mode='w') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(landmarks)

# write_landmarks()

cap.release()
cv2.destroyAllWindows()