import cv2

video=cv2.VideoCapture(0)

facedetect=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

count=0

while True:
	ret,frame=video.read()
	faces=facedetect.detectMultiScale(frame,1.0, 3)
	for x,y,w,h in faces:
		count=count+1
		name='./dataset/with_mask/'+ str(count) + '.png'
		print("Creating Images........." +name)
		cv2.imwrite(name, frame[y:y+h,x:x+w])
		cv2.rectangle(frame, (x,y), (x+w, y+h), (0,222,0), 3)
	cv2.imshow("WindowFrame", frame)
	cv2.waitKey(1)
	if count>1:
		break
video.release()
cv2.destroyAllWindows()





