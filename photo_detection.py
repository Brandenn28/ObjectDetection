import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input

# Load the YOLOv4 model
net = cv2.dnn.readNetFromDarknet('yolov4_custom.cfg', 'yolov4.weights')

# Load the trained model
model = tf.keras.models.load_model('twoStep.h5')

# Load the labels for the classes
class_labels = ['Not Fallen', 'Fall']

# Load the photo
photo_path = 'test2.jpg'
image = cv2.imread(photo_path)

# Get the dimensions of the image
height, width, _ = image.shape

# Create a blob from the image to feed into YOLOv4
blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)

# Set the input for YOLOv4
net.setInput(blob)

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
outs = net.forward(output_layers)


# Process the outputs of YOLOv4
class_ids = []
confidences = []
boxes = []

for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]

        if confidence > 0.5:  # Consider all detected objects with confidence above the threshold
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            bbox_width = int(detection[2] * width)
            bbox_height = int(detection[3] * height)

            x = int(center_x - bbox_width / 2)
            y = int(center_y - bbox_height / 2)

            class_ids.append(class_id)
            confidences.append(float(confidence))
            boxes.append([x, y, bbox_width, bbox_height])

# Sort the detections by confidence in descending order
indices = np.argsort(confidences)[::-1]
sorted_boxes = [boxes[i] for i in indices]

# Perform classification on the most confident detection
x, y, w, h = sorted_boxes[0]

# Crop the region of interest (ROI) containing the human
roi = image[y:y+h, x:x+w]

# Resize the ROI to match the input size of the classification model
resized_roi = cv2.resize(roi, (416, 416))

# Preprocess the ROI
preprocessed_roi = preprocess_input(resized_roi)

# Add an extra dimension to match the expected input shape of the classification model
input_roi = np.expand_dims(preprocessed_roi, axis=0)

# Make predictions on the input ROI
predictions = model.predict(input_roi)

# Get the predicted class label and its probability
predicted_class = np.argmax(predictions[0])
predicted_label = class_labels[predicted_class]
confidence = predictions[0][predicted_class]

# Display the predicted label and confidence on the image
label_text = f'{predicted_label} ({confidence:.2f})'
cv2.putText(image, label_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

# Add a bounding box around the detected human
cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Display the image with the bounding box and predicted label
cv2.imshow('Object Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
