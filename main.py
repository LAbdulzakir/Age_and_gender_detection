import cv2
import dlib

# Load the pre-trained models
face_detector = dlib.get_frontal_face_detector()
gender_classifier = cv2.dnn.readNetFromCaffe(
    'gender_deploy.prototxt', 'gender_net.caffemodel')
age_classifier = cv2.dnn.readNetFromCaffe(
    'age_deploy.prototxt', 'age_net.caffemodel')

# Define the list of age and gender labels
age_labels = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)',
              '(38-43)', '(48-53)', '(60-100)']
gender_labels = ['Male', 'Female']

# Load the input image
image = cv2.imread('input_image.jpg')

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the grayscale image
faces = face_detector(gray)

# Iterate over detected faces
for face in faces:
    # Extract the face region of interest (ROI)
    x, y, w, h = face.left(), face.top(), face.width(), face.height()

    # Preprocess the ROI for gender classification
    face_blob = cv2.dnn.blobFromImage(image[y:y + h, x:x + w], 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)

    # Feed the face blob to the gender classifier
    gender_classifier.setInput(face_blob)
    gender_predictions = gender_classifier.forward()

    # Get the gender label with the highest probability
    gender_index = gender_predictions[0].argmax()
    gender_label = gender_labels[gender_index]

    # Preprocess the ROI for age classification
    face_blob = cv2.dnn.blobFromImage(image[y:y + h, x:x + w], 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)

    # Feed the face blob to the age classifier
    age_classifier.setInput(face_blob)
    age_predictions = age_classifier.forward()

    # Get the age label with the highest probability
    age_index = age_predictions[0].argmax()
    age_label = age_labels[age_index]

    # Draw the bounding box and labels on the image
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    label = f'{gender_label}, {age_label}'
    cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

# Display the output image
cv2.imshow('Gender and Age Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
