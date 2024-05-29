from ultralytics import YOLO
import cv2

# Load the model
model = YOLO(".\\runs\\segment\\detector_model\\weights\\last.pt")

# Initialize video capture from the default camera (usually the first one)
cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    print("Error: Could not open video source.")
    exit()

# Loop to continuously get frames and process them
while True:
    ret, frame = cap.read()  # Capture frame-by-frame
    if not ret:
        print("Error: Could not read frame.")
        break

    # Ensure the frame is not flipped
    frame = cv2.flip(frame, 1)  # Flip the frame to correct the mirror effect

    results = model.predict(source=frame, show=False)  # Run prediction on the frame

    # Draw results on the frame
    annotated_frame = results[0].plot()  # Assuming results is a list and we want the first result

    cv2.imshow('YOLO Detection', annotated_frame)  # Display the resulting frame

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Close the window when 'q' key is pressed
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

