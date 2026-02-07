import cv2
import numpy as np
from tensorflow.keras.models import load_model


model = load_model('drone_model.h5')


labels = ['BACKWARD', 'FLIP', 'FORWARD', 'HOVER', 'MOVE_DOWN', 'MOVE_UP', 'START', 'STOP']

cap = cv2.VideoCapture(0)

print("Starting Webcam... Press space to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # To give mirror effect
    frame = cv2.flip(frame, 1)


    # Resize to 128x128 to match your training input
    img = cv2.resize(frame, (128, 128))
    img = img / 255.0  # Normalize pixels
    img = np.expand_dims(img, axis=0) # Add batch dimension

    #Prediction
    prediction = model.predict(img, verbose=0)
    class_idx = np.argmax(prediction)
    confidence = prediction[0][class_idx]

    # result
    # We only show the command if the model is more than 80% sure
    if confidence > 0.80:
        current_cmd = labels[class_idx]
        display_text = f"COMMAND: {current_cmd} ({confidence*100:.1f}%)"
        color = (0, 255, 0) # Green for high confidence
    else:
        display_text = "Waiting for gesture..."
        color = (0, 0, 255) # Red for low confidence

    cv2.putText(frame, display_text, (10, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

    cv2.imshow('Drone Gesture Control System', frame)

    if cv2.waitKey(1) & 0xFF == ord(' '):
        break

cap.release()
cv2.destroyAllWindows()
