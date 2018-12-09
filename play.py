import cv2
import numpy as np
from keras.models import load_model
import pickle


model = load_model('model.h5')



file = open("train_data.pkl","rb")
signs = pickle.load(file)
file.close()

def predict(hand):
    img = cv2.resize(hand, (50,50) )
    img = np.array(img)
    img = img.reshape( (1,50,50,1) )
    img = img/255.0
    res = model.predict( img )
    max_ind = res.argmax()
    return gesture[ max_ind ]

def process(img):
	hand = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	blur = cv2.GaussianBlur(hand, (11,11), 0)
	blur = cv2.medianBlur(blur, 15)
	thresh = cv2.threshold(blur,210,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
	thresh = cv2.bitwise_not(thresh)
	return thresh

vc = cv2.VideoCapture(0)


image_x_left = 50
image_y_left = 100
image_w = 200
image_h = 200
image_x_right = 400
image_y_right = 100



while True:
    
	rval, frame = vc.read()

	if not rval:
		continue
        
	frame = cv2.flip(frame, 1)
        
	cv2.rectangle(frame, (image_x_left,image_y_left), (image_x_left + image_w,image_y_left + image_h), (0,255,0), 2)
	cv2.rectangle(frame, (image_x_right,image_y_right), (image_x_right + image_w,image_y_right + image_h), (0,255,0), 2)

	left_hand = frame[image_y_left:image_y_left+image_h, image_x_left:image_x_left+image_w]
	left_thresh = process(left_hand)

	right_hand = frame[image_y_right:image_y_right+image_h, image_x_right:image_x_right+image_w]
	right_thresh = process(right_hand)

	cv2.imshow("image", frame)
	cv2.imshow("Left Hand", left_thresh)
	cv2.imshow("Right Hand", right_thresh)
        
	keypress = cv2.waitKey(1)
    
	if keypress == ord('q'):
		break

vc.release()
cv2.destroyAllWindows()