import cv2
import numpy as np
import pyttsx3
import time

# Load the pre-trained MobileNet SSD model
net = cv2.dnn.readNetFromCaffe('MobileNetSSD_deploy.prototxt', 'MobileNetSSD_deploy.caffemodel')

# Define the classes for the objects the model can detect
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair",
           "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

# Set confidence threshold for object detection
CONFIDENCE_THRESHOLD = 0.2

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Open a video capture object (0 is the default camera)
cap = cv2.VideoCapture(0)

# Initialize a timer for speaking intervals
last_speak_time = time.time()
speak_interval = 5  # Set the desired time interval in seconds

while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    # Check if the frame is valid
    if not ret:
        print("Error: Couldn't read frame. Exiting...")
        break

    # Resize the frame to a fixed width for faster processing
    frame = cv2.resize(frame, (500, 500))

    # Get the height and width of the frame
    (h, w) = frame.shape[:2]

    # Preprocess the image for object detection
    blob = cv2.dnn.blobFromImage(frame, 0.007843, (500, 500), 127.5)

    # Set the input to the pre-trained model
    net.setInput(blob)

    # Run forward pass and get the output
    detections = net.forward()

    # Loop over the detections
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        # Check if the detection meets the confidence threshold
        if confidence > CONFIDENCE_THRESHOLD:
            class_id = int(detections[0, 0, i, 1])

            # Get the coordinates of the bounding box
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # Get the label of the detected object
            label = CLASSES[class_id]

            # Check if it's time to speak
            current_time = time.time()
            if current_time - last_speak_time >= speak_interval:
                # Speak the name of the object using text-to-speech
                engine.say(label)
                engine.runAndWait()

                # Update the last speak time
                last_speak_time = current_time

            # Draw the bounding box and label on the frame
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            label = "{}: {:.2f}%".format(label, confidence * 100)
            cv2.putText(frame, label, (startX, startY - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow("Object Detection", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close the OpenCV windows
cap.release()
cv2.destroyAllWindows()
