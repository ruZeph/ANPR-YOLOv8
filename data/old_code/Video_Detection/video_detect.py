import os

import cv2
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('weights/best.pt')

# Open the video file
video_path = "video_sample/video_20230408_135100.mp4"
video_output = "video_outputs/"

if not os.path.exists(video_output):
    os.makedirs(video_output)

# Capture video
cap = cv2.VideoCapture(video_path)

# For output
title = os.path.join(video_output, 'output_video.avi')
print(title)
cap_out = cv2.VideoWriter(title, cv2.VideoWriter_fourcc(*'XVID'), cap.get(cv2.CAP_PROP_FPS), (1280, 720))

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Resize the frame
        resize = cv2.resize(frame, (1280, 720))
        # print(resize.shape[0], resize.shape[1])

        # Run YOLOv8 inference on the frame
        results = model.predict(resize)

        # # Visualize the results on the frame
        annotated_frame = results[0].plot()

        cap_out.write(annotated_frame)

        # Display the annotated frame
        cv2.imshow("YOLOv8 Inference", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# # Release the video capture object and close the display window
cap.release()
cap_out.release()
cv2.destroyAllWindows()
