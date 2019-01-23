#i want see all images and turn them into trainable data

import cv2
import os
from PIL import Image
import numpy as np
import pickle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR,'images')

faceCascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_default.xml')

#face recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

currentID = 0
label_ids = {}  #dict 
y_labels = []
x_train = []


for root,dirs,files in os.walk(image_dir):
	for file in files:
		if file.endswith('png') or file.endswith('jpg') or file.endswith('JPG'):
			path = os.path.join(root,file)
			label = os.path.basename(os.path.dirname(path)).replace(" ","-").lower()
									#we can change to root as well
			if label not in label_ids:
				label_ids[label] = currentID
				currentID += 1
			id_ = label_ids[label]
			# y_labels.append(label) #we need to store some number instead
			# x_train.append(path) #verify this image, turn into a NUMPY array and GRAY color

			#turning image to numpy array using PIL (python image library), rmb to install pillow (pip install pillow --upgrade)
			pil_image = Image.open(path).convert("L")  #"L" turns into gray
			
			#optimized for resize
			size = (550,550)
			final_image = pil_image.resize(size, Image.ANTIALIAS)

			image_array = np.array(final_image,'uint8')  #uint8 unsinged integer 0 to 255 and converted it to numpy array
			faces = faceCascade.detectMultiScale(image_array)#,scaleFactor=1.1,minNeighbors=5)

			for (x,y,w,h) in faces:
				roi = image_array[y:y+h,x:x+w]
				x_train.append(roi)   #now we have our training data
				y_labels.append(id_)   #we store each face with assigned label

with open("labels.pickle", 'wb') as f: # wb means write bytes as f (files)
	pickle.dump(label_ids,f)   #dump labels id to files becuase when run prediction, we want to match the face with the label


recognizer.train(x_train,np.array(y_labels))  #we need to change ylabels to np array because
recognizer.save("trainner.yml")