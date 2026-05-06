import cv2
import numpy as np
import torch
from torch import nn
from torchvision import models
from PIL import Image
from collections import deque, Counter


class_names = ['Colors', 'Darks', 'Whites'] 
num_classes = 3

print("Loading AI Model...")
model = models.resnet50(weights=None) 
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, num_classes)

model.load_state_dict(torch.load('new_laundry_resnet50.pth', map_location=torch.device('cpu'), weights_only=True))
model.eval() 

preprocess = models.ResNet50_Weights.DEFAULT.transforms()


cap = cv2.VideoCapture(0)
current_mode = None  


prediction_buffer = deque(maxlen=15)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not access the webcam.")
        break
        
    frame = cv2.flip(frame, 1)
    height, width, _ = frame.shape
    

    if current_mode is None:
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (width, height), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)

        cv2.putText(frame, "AI LAUNDRY SORTER", (width//2 - 150, height//2 - 80), cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 255, 255), 2)
        cv2.putText(frame, "Select Washing Mode to Begin:", (width//2 - 190, height//2 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        
        cv2.putText(frame, "[ W ] - Whites Only", (width//2 - 120, height//2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, "[ D ] - Darks Only", (width//2 - 120, height//2 + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 2)
        cv2.putText(frame, "[ C ] - Colors Only", (width//2 - 120, height//2 + 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, "Press 'Q' to Quit", (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)


    else:
        box_size = 250
        x1 = int(width / 2 - box_size / 2)
        y1 = int(height / 2 - box_size / 2)
        x2 = x1 + box_size
        y2 = y1 + box_size

        roi = frame[y1:y2, x1:x2]

        rgb_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_roi)
        batch = preprocess(pil_img).unsqueeze(0)

        with torch.no_grad():
            prediction = model(batch).squeeze(0).softmax(0)
            class_id = prediction.argmax().item()
            score = prediction[class_id].item()
            
        raw_class = class_names[class_id]
        conf_percent = score * 100

        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        texture_variance = np.var(gray_roi)

        is_empty_background = texture_variance < 150  

        if is_empty_background or conf_percent < 45.0:
            prediction_buffer.append("Unsure")
        else:
            prediction_buffer.append(raw_class)
            
        smoothed_class = Counter(prediction_buffer).most_common(1)[0][0]

        box_color = (255, 255, 255) 
        status_text = ""

        if smoothed_class == "Unsure":
            box_color = (200, 200, 200)
            status_text = "Waiting for item..."
        else:
            if current_mode == 'Whites':
                if smoothed_class == 'Whites':
                    box_color, status_text = (0, 255, 0), "SAFE: OK to wash!"
                else:
                    box_color, status_text = (0, 0, 255), f"WARNING: {smoothed_class.upper()} DETECTED!"
                    
            elif current_mode == 'Darks':
                if smoothed_class == 'Darks':
                    box_color, status_text = (0, 255, 0), "SAFE: OK to wash!"
                else:
                    box_color, status_text = (0, 0, 255), f"WARNING: {smoothed_class.upper()} DETECTED!"

            elif current_mode == 'Colors':
                if smoothed_class == 'Colors':
                    box_color, status_text = (0, 255, 0), "SAFE: OK to wash!"
                else:
                    box_color, status_text = (0, 0, 255), f"WARNING: {smoothed_class.upper()} DETECTED!"

        bleed_risk_alert = False
        avg_saturation = 0 
        avg_brightness = 0
        
        if smoothed_class == 'Colors':
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            avg_saturation = np.mean(hsv[:, :, 1])
            avg_brightness = np.mean(hsv[:, :, 2]) 
            

            if avg_saturation > 80 and avg_brightness < 200: 
                bleed_risk_alert = True
        

        hud_bg = frame.copy()
        cv2.rectangle(hud_bg, (0, 0), (width, 100), (0, 0, 0), -1)
        frame = cv2.addWeighted(hud_bg, 0.7, frame, 0.3, 0)

        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 3)
        cv2.putText(frame, "Fill box with clothing", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1)

        cv2.putText(frame, f"MODE: {current_mode.upper()}", (20, 35), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 200, 0), 2)
        cv2.putText(frame, status_text, (20, 75), cv2.FONT_HERSHEY_DUPLEX, 0.8, box_color, 2)

        top_right_x = width - 260
        if smoothed_class == "Unsure":
            cv2.putText(frame, "AI: Unsure", (top_right_x, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        else:
            cv2.putText(frame, f"AI: {smoothed_class}", (top_right_x, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Conf: {conf_percent:.1f}%", (top_right_x, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        if bleed_risk_alert:
            cv2.putText(frame, "BLEED RISK: Wash Separately!", (20, height - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 3)

        cv2.putText(frame, "Press 'R' to Reselect Mode | 'Q' to Quit", (20, height - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    cv2.imshow("Smart Laundry Sorter", frame)


    key = cv2.waitKey(1) & 0xFF
    if key == ord('w'):
        current_mode = 'Whites'
        prediction_buffer.clear() 
    elif key == ord('d'):
        current_mode = 'Darks'
        prediction_buffer.clear()
    elif key == ord('c'):
        current_mode = 'Colors'
        prediction_buffer.clear()
    elif key == ord('r'):
        current_mode = None  
        prediction_buffer.clear()
    elif key == ord('q'): 
        break

cap.release()
cv2.destroyAllWindows()