# face-recognition
Face recognition find faces and their labels
# Features
Find all the faces that appear in a picture:

import face_recognition
image = face_recognition.load_image_file("your_file.jpg")
face_locations = face_recognition.face_locations(image)
