# face-recognition
Face recognition find faces and their labels
# Features
Find all the faces that appear in a picture:

import face_recognition
image = face_recognition.load_image_file("your_file.jpg")
face_locations = face_recognition.face_locations(image)

# Files needed
1- harcasecade file i have uploaded 
2- tester.py file
3- face recognition file
4- create TestImages folder
5- create trainingImage folder under it create two folders 0,1 named for providing labels and training data
