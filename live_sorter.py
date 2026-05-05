import cv2
import numpy as np
import torch
from torch import nn
from torchvision import models
from PIL import Image


class_names = ['Colors', 'Darks', 'Whites'] 
num_classes = 3


model = models.resnet50(weights=None) 
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, num_classes)


model.load_state_dict(torch.load('new_laundry_resnet50.pth', map_location=torch.device('cpu'), weights_only=True))
model.eval() 


preprocess = models.ResNet50_Weights.DEFAULT.transforms()

cap = cv2.VideoCapture(0)
current_mode = "None"
print("App Started! Press 'W' for Whites, 'D' for Darks, 'C' for Colors. 'Q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break
        

    frame = cv2.flip(frame, 1)
    

    height, width, _ = frame.shape
    

    box_size = 250
    x1 = int(width / 2 - box_size / 2)
    y1 = int(height / 2 - box_size / 2)
    x2 = x1 + box_size
    y2 = y1 + box_size
    

    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
    cv2.putText(frame, "Place Item Here", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

 
    roi = frame[y1:y2, x1:x2]


    rgb_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb_roi)
    
    batch = preprocess(pil_img).unsqueeze(0)

    with torch.no_grad():
        prediction = model(batch).squeeze(0).softmax(0)
        class_id = prediction.argmax().item()
        score = prediction[class_id].item()
        
    predicted_class = class_names[class_id]
    conf_percent = score * 100


    bleed_risk_alert = False
    if predicted_class == 'Colors':
  
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        avg_saturation = np.mean(hsv[:, :, 1])
        

        if avg_saturation > 180:
            bleed_risk_alert = True


    display_text = f"Detected: {predicted_class} ({conf_percent:.1f}%)"
    text_color = (255, 255, 255) 
    alert_triggered = False

    if current_mode == 'Whites':
        if predicted_class == 'Whites':
            display_text = "Safe [Whites Mode]"
            text_color = (0, 255, 0) 
        else:
            display_text = "ALERT: NOT WHITE! REMOVE!"
            alert_triggered = True

    elif current_mode == 'Darks':
        if predicted_class == 'Darks':
            display_text = "Safe[Darks Mode]"
            text_color = (0, 255, 0)
        else:
            display_text = "ALERT: NOT DARK! REMOVE!"
            alert_triggered = True

    elif current_mode == 'Colors':
        if predicted_class == 'Colors':
            display_text = "Safe [Colors Mode]"
            text_color = (0, 255, 0)
        else:
            display_text = "ALERT: INCORRECT LOAD!"
            alert_triggered = True


    if alert_triggered:
        red_screen = np.zeros_like(frame)
        red_screen[:, :, 2] = 255 
        frame = cv2.addWeighted(frame, 0.6, red_screen, 0.4, 0)
        print("\a") 


    if bleed_risk_alert:
        cv2.putText(frame, "WARNING: HIGH PIGMENT DENSITY!", (10, 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
        cv2.putText(frame, "Likely to bleed. Wash alone first time.", (10, 110), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

    cv2.putText(frame, f"Mode: {current_mode}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    cv2.putText(frame, display_text, (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2)


    cv2.imshow("Smart Laundry Sorter", frame)


    key = cv2.waitKey(1) & 0xFF
    if key == ord('w'):
        current_mode = 'Whites'
    elif key == ord('d'):
        current_mode = 'Darks'
    elif key == ord('c'):
        current_mode = 'Colors'
    elif key == ord('q'): 
        break

cap.release()
cv2.destroyAllWindows()