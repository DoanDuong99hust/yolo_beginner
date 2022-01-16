import cv2
import dlib
import math
import time
import numpy as np
from imutils.video import FPS
import imutils

# car_detect = cv2.CascadeClassifier('car_detect_harrcascade.xml')
video = cv2.VideoCapture('test_video/30km2.mp4')

fs = video.get(cv2.CAP_PROP_FPS)

writer = None
(W, H) = (None, None)

# try to determine the total number of frames in the video file
try:
    prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
        else cv2.CAP_PROP_FRAME_COUNT
    total = int(video.get(prop))
    print("[INFO] {} total frames in video".format(total))

# an error occurred while trying to determine the total
# number of frames in the video file
except:
    print("[INFO] could not determine # of frames in video")
    print("[INFO] no approx. completion time can be provided")
    total = -1


# Dinh nghia cac tham so dai , rong
f_width = 1280
f_height = 720

# Cai dat tham so : so diem anh / 1 met, o day dang de 1 pixel = 1 met
pixels_per_meter = 6

# Cac tham so phuc vu tracking
frame_idx = 0
car_number = 0
fps = 0
writer = None
carTracker = {}
carNumbers = {}
carStartPosition = {}
carCurrentPosition = {}
speed = [None] * 1000


# Ham xoa cac tracker khong tot
def remove_bad_tracker():
	global carTracker, carStartPosition, carCurrentPosition

	# Xoa cac car tracking khong tot
	delete_id_list = []

	# Duyet qua cac car
	for car_id in carTracker.keys():
		# Voi cac car ma conf tracking < 4 thi dua vao danh sach xoa
		if carTracker[car_id].update(image) < 6:
			delete_id_list.append(car_id)

	# Thuc hien xoa car
	for car_id in delete_id_list:
		carTracker.pop(car_id, None)
		carStartPosition.pop(car_id, None)
		carCurrentPosition.pop(car_id, None)

	return

# Ham tinh toan toc do
def calculate_speed(startPosition, currentPosition, fps):

	global pixels_per_meter

	# Tinh toan khoang cach di chuyen theo pixel
	distance_in_pixels = math.sqrt(math.pow(currentPosition[0] - startPosition[0], 2) + math.pow(currentPosition[1] - startPosition[1], 2))

	# Tinh toan khoang cach di chuyen bang met
	distance_in_meters = distance_in_pixels / pixels_per_meter

	# Tinh toc do met tren giay
	speed_in_meter_per_second = distance_in_meters * fps
	# Quy doi sang km/h
	speed_in_kilometer_per_hour = speed_in_meter_per_second * 3.6

	return speed_in_kilometer_per_hour

def get_output_layers(net):
    layer_names = net.getLayerNames()

    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers



weights = "yolov3.weights"
config = "yolov3.cfg"
def get_object(weights,config,image,):
	net = cv2.dnn.readNet(weights, config)
	Width = image.shape[1]
	Height = image.shape[0]
	scale = 0.00392
	net = cv2.dnn.readNet(weights, config)
	blob = cv2.dnn.blobFromImage(image, scale, (416, 416), (0, 0, 0), True, crop=False)
	net.setInput(blob)
	outs = net.forward(get_output_layers(net))
	class_ids = []
	confidences = []
	boxes = []
	conf_threshold = 0.5
	nms_threshold = 0.4
	for out in outs:
		for detection in out:
			scores = detection[5:]
			class_id = np.argmax(scores)
			confidence = scores[class_id]
			if confidence > 0.5:
				center_x = int(detection[0] * Width)
				center_y = int(detection[1] * Height)
				w = int(detection[2] * Width)
				h = int(detection[3] * Height)
				x = center_x - w / 2
				y = center_y - h / 2
				class_ids.append(class_id)
				confidences.append(float(confidence))
				boxes.append([x, y, w, h])
	indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
	selectbox= []
	for i in indices:
		# i = i[0]
		
		box = boxes[i]
		x = box[0]
		y = box[1]
		w = box[2]
		h = box[3]
		selectbox.append(box)
	return selectbox

while True:
	start_time = time.time()
	_, image = video.read()
	
	if image is None:
		break

	image = cv2.resize(image, (f_width, f_height))
	output_image = image.copy()

	frame_idx += 1
	remove_bad_tracker()

 	# Thuc hien detect moi 10 frame
	if not (frame_idx % 30):

		# Thuc hien detect car trong hinh
		# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		# cars = car_detect.detectMultiScale(gray, 1.2, 13, 18, (24, 24))
		cars = get_object(weights,config,image)
		# Duyet qua cac car detect duoc
		for (_x, _y, _w, _h) in cars:
			x = int(_x)
			y = int(_y)
			w = int(_w)
			h = int(_h)

			# Tinh tam diem cua car
			x_center = x + 0.5 * w
			y_center = y + 0.5 * h


			matchCarID = None
			# Duyet qua cac car da track
			for carID in carTracker.keys():
				# Lay vi tri cua car da track
				trackedPosition = carTracker[carID].get_position()
				t_x = int(trackedPosition.left())
				t_y = int(trackedPosition.top())
				t_w = int(trackedPosition.width())
				t_h = int(trackedPosition.height())
				# Tinh tam diem cua car da track
				t_x_center = t_x + 0.5 * t_w
				t_y_center = t_y + 0.5 * t_h

				# Kiem tra xem co phai ca da track hay khong
				if (t_x <= x_center <= (t_x + t_w)) and (t_y <= y_center <= (t_y + t_h)) and (x <= t_x_center <= (x + w)) and (y <= t_y_center <= (y + h)):
					matchCarID = carID

			# Neu khong phai car da track thi tao doi tuong tracking moi
			if matchCarID is None:

				tracker = dlib.correlation_tracker()
				tracker.start_track(image, dlib.rectangle(x, y, x + w, y + h))

				carTracker[car_number] = tracker
				carStartPosition[car_number] = [x, y, w, h]

				car_number +=1

	# Thuc hien update position cac car
	for carID in carTracker.keys():
		trackedPosition = carTracker[carID].get_position()

		t_x = int(trackedPosition.left())
		t_y = int(trackedPosition.top())
		t_w = int(trackedPosition.width())
		t_h = int(trackedPosition.height())

		cv2.rectangle(output_image, (t_x, t_y), (t_x + t_w, t_y + t_h), (255,0,0), 4)
		carCurrentPosition[carID] = [t_x, t_y, t_w, t_h]

	# Tinh toan frame per second
	end_time = time.time()
	if not (end_time == start_time):
		fps = 1.0/(end_time - start_time)

	# Lap qua cac xe da track va tinh toan toc do
	for i in carStartPosition.keys():
		[x1, y1, w1, h1] = carStartPosition[i]
		[x2, y2, w2, h2] = carCurrentPosition[i]
		p0 = (int(x2 + (w2-x2)/2), int(y2 + (h2-y2)/2))

		carStartPosition[i] = [x2, y2, w2, h2]
		
		# Neu xe co di chuyen thi
		if [x1, y1, w1, h1] != [x2, y2, w2, h2]:
			# Neu nhu chua tinh toan toc do va toa do hien tai < 200 thi tinh toan toc do
			if (speed[i] is None or speed[i] == 0):
				speed[i] = calculate_speed([x1, y1, w1, h1], [x2, y2, w2, h2], fs)

			# Neu nhu da tinh toc do va xe da vuot qua tung do 200 thi hien thi tong do
			if speed[i] is not None and speed[i] > 0.0:
				# if speed[i] > 80:
				# 	cv2.putText(output_image, str (int(speed[i])-80) + " km/h vuot qua",
				# 			(x2, y2), cv2.FONT_HERSHEY_SIMPLEX, 1,
				# 			(0, 0, 255), 2)
				# else:			   
				cv2.putText(output_image, str (int(speed[i])) + " km/h",
						(x2+w2,  y2),cv2.FONT_HERSHEY_SIMPLEX, 1,
						(0, 255, 255), 2)
			# if speed[i] >= 80:
			#   	cv2.putText(output_image, str(int(speed[i])) + " km/h LIMIT",
			#    				(x2, y2), cv2.FONT_HERSHEY_SIMPLEX, 1,
			#    				(0, 0, 255), 2) 
			print("car key: ",i,"speed",speed[i],"fs: ", fs)
	if writer is None:
		fourcc = cv2.VideoWriter_fourcc(*'mp4v')
		writer = cv2.VideoWriter('result-highway.mp4', fourcc, 30,
                                 (output_image.shape[1], output_image.shape[0]), True)
		writer.write(output_image)
    	
	cv2.imshow('video', output_image)
	#Detect phim Q
	if cv2.waitKey(1) == ord('q'):
		break

	# fps.update()

# fps.stop()

print("[INFO] totalFrame: {:.2f}".format(frame_idx))
# print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps))

cv2.destroyAllWindows()
