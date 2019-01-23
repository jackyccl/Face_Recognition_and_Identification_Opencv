import cv2
import numpy as numpy
import pickle

faceCascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainner.yml')
cap = cv2.VideoCapture(0)

labels = {}
with open('labels.pickle','rb') as f:
	org_labels = pickle.load(f)
	labels = {i:j for j,i in org_labels.items()}

while(True):

	ret, frame = cap.read() 		#Capture frame-by-frame
	gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
	faces = faceCascade.detectMultiScale(
							gray,
							scaleFactor=1.5,
							minNeighbors=5
			)

	for x,y,w,h in faces:   #ROI - region of interest
		#print(x,y,w,h)
		roi_gray = gray[y:y+h, x:x+w]  
		roi_color = frame[y:y+h,x:x+w]

		#how do we recognize ROI (who that person is)? where we can also use deep learned model, keras tensorflow pytorch scikitlearn
		id_, conf = recognizer.predict(roi_gray)  #labels and confidence od detection
		print(conf)
		if conf <=60:
			print(id_)
			font = cv2.FONT_HERSHEY_SIMPLEX
			name = labels[id_]
			color = (255,255,255)
			stroke = 2
			cv2.putText(frame,name,(x,y),font,1,color,stroke,cv2.LINE_AA)


		saved_image = 'my_image.png'
		cv2.imwrite(saved_image, roi_gray)

		color_square = (0,255,0) #BGR
		stroke = 2 				 #line thickness
		coor_end_x = x+w
		coor_end_y = y+h
		cv2.rectangle(frame,(x,y),(coor_end_x,coor_end_y),color_square,stroke)


	cv2.imshow('Frame',frame)		#display the resulting frame
	if cv2.waitKey(20) & 0xFF == ord('q'):
		break


cap.release()			#release capture when done
cv2.destroyAllWindows()




'''

This "detectMultiScale" function detects the actual face and is the key part of our code, so let’s go over the options:

1. The detectMultiScale function is a general function that detects objects. Since we are calling it on the face cascade, that’s what it detects.

2. The first option is the grayscale image.

3. The second is the scaleFactor. Since some faces may be closer to the camera, they would appear bigger than the faces in the back. The scale factor compensates for this.

4. The detection algorithm uses a moving window to detect objects. minNeighbors defines how many objects are detected near the current one before it declares the face found. minSize, meanwhile, gives the size of each window.

'''