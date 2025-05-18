import cv2
import mediapipe as mp
import pyautogui
import numpy as np

pyautogui.FAILSAFE = True

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

screen_w, screen_h = pyautogui.size()
click_threshold = 40  # Distance threshold for pinch
margin = 10  # Prevent hitting screen corners

dragging = False  # Track if we’re currently dragging

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            index_tip = hand_landmarks.landmark[8]
            thumb_tip = hand_landmarks.landmark[4]

            h, w, _ = img.shape
            x_index, y_index = int(index_tip.x * w), int(index_tip.y * h)
            x_thumb, y_thumb = int(thumb_tip.x * w), int(thumb_tip.y * h)

            pinch_distance = np.sqrt((x_thumb - x_index) ** 2 + (y_thumb - y_index) ** 2)

            offset_x = (index_tip.x - 0.5) / 0.4
            offset_y = (index_tip.y - 0.5) / 0.4

            offset_x = max(min(offset_x, 1), -1)
            offset_y = max(min(offset_y, 1), -1)

            screen_x = int((offset_x + 0.5) * screen_w)
            screen_y = int((offset_y + 0.5) * screen_h)

            screen_x = max(min(screen_x, screen_w - margin), margin)
            screen_y = max(min(screen_y, screen_h - margin), margin)

            # Move the cursor (while dragging or not)
            pyautogui.moveTo(screen_x, screen_y)

            # Drag and drop logic
            if pinch_distance < click_threshold:
                if not dragging:
                    pyautogui.mouseDown()
                    dragging = True
            else:
                if dragging:
                    pyautogui.mouseUp()
                    dragging = False

            # Visualization
            cv2.circle(img, (x_index, y_index), 10, (0, 255, 255), cv2.FILLED)
            cv2.circle(img, (x_thumb, y_thumb), 10, (0, 255, 0), cv2.FILLED)

    cv2.imshow("Hand Tracking - Drag and Drop", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
