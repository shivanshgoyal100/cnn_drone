import cv2
import os

# Map filenames to labels based on your requirement
gesture_mapping = {
    'stop.MP4': 'STOP',          # Closed fist -> Stop
    'start.MP4': 'START',        # 5 fingers -> Start
    'forward.MP4': 'FORWARD',    # 1 finger -> Move Forward
    'backward.MP4': 'BACKWARD',  # 2 fingers -> Move Backward
    'flip.MP4': 'FLIP',          # Thumb + index -> Flip
    'up.MP4': 'MOVE_UP',    # Thumb up -> Move Up
    'down.MP4': 'MOVE_DOWN',# 3 fingers -> Move Down
    'hover.MP4': 'HOVER'         # O gesture -> Hover
}

def extract():
    for video_file, label in gesture_mapping.items():
        video_path = os.path.join('right', video_file)
        save_path = os.path.join('dataset', label)
        
        # This part adds the folders automatically
        os.makedirs(save_path, exist_ok=True)
        
        cap = cv2.VideoCapture(video_path)
        count = 0
        while True:
            ret, frame = cap.read()
            if not ret: break
            cv2.imwrite(os.path.join(save_path, f"{count}.jpg"), frame)
            count += 1
        cap.release()
        print(f"Created {count} images in dataset/{label}")

if __name__ == "__main__":
    extract()
