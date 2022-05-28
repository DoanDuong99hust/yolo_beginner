import cv2
import dlib
import math
import time
import numpy as np

# car_detect = cv2.CascadeClassifier('car_detect_harrcascade.xml')
video = cv2.VideoCapture('video.mp4')

# Dinh nghia cac tham so dai , rong
f_width = 1280
f_height = 720

line1 = np.array([(665, 2), (663, 719)])
line2 = np.array([(712, 3), (1071, 719)])
line3 = np.array([(617, 3), (257, 719)])
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
    distance_in_pixels = math.sqrt(
        math.pow(currentPosition[0] - startPosition[0], 2) + math.pow(currentPosition[1] - startPosition[1], 2))

    # Tinh toan khoang cach di chuyen bang met
    distance_in_meters = distance_in_pixels / pixels_per_meter

    # Tinh toc do met tren giay
    speed_in_meter_per_second = distance_in_meters * fps
    # Quy doi sang km/h
    speed_in_kilometer_per_hour = speed_in_meter_per_second * 3.6

    return speed_in_kilometer_per_hour


def get_output_layers(net):
    layer_names = net.getLayerNames()

    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers


weights = "yolov3.weights"
config = "yolov3.cfg"


def get_object(weights, config, image, ):
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
    selectbox = []
    for i in indices:
        i = i[0]

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

    lane3 = np.array([[[997, 0], [1068, 0], [1607, 1080], [995, 1080]]], np.int32)
    lane2 = np.array([[[926, 0], [997, 0], [995, 1080], [386, 1080]]], np.int32)
    lane1 = np.array([[[870, 0], [926, 0], [386, 1080], [0, 836]]], np.int32)
    lane4 = np.array([[[1068, 0], [1155, 0], [1920, 711], [1607, 1080]]], np.int32)
    cv2.polylines(image, [lane1], True, (0, 255, 0), thickness=1)
    cv2.polylines(image, [lane2], True, (0, 0, 255), thickness=1)
    cv2.polylines(image, [lane3], True, (0, 255, 0), thickness=1)
    cv2.polylines(image, [lane4], True, (0, 0, 255), thickness=1)

    if image is None:
        break

    # image = cv2.resize(image, (f_width, f_height))
    output_image = image.copy()

    frame_idx += 1
    remove_bad_tracker()

    # Thuc hien detect moi 10 frame
    if not (frame_idx % 10):

        # Thuc hien detect car trong hinh
        # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # cars = car_detect.detectMultiScale(gray, 1.2, 13, 18, (24, 24))
        cars = get_object(weights, config, image)
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
                if (t_x <= x_center <= (t_x + t_w)) and (t_y <= y_center <= (t_y + t_h)) and (
                        x <= t_x_center <= (x + w)) and (y <= t_y_center <= (y + h)):
                    matchCarID = carID

            # Neu khong phai car da track thi tao doi tuong tracking moi
            if matchCarID is None:
                tracker = dlib.correlation_tracker()
                tracker.start_track(image, dlib.rectangle(x, y, x + w, y + h))

                carTracker[car_number] = tracker
                carStartPosition[car_number] = [x, y, w, h]

                car_number += 1

    # Thuc hien update position cac car
    for carID in carTracker.keys():
        trackedPosition = carTracker[carID].get_position()

        t_x = int(trackedPosition.left())
        t_y = int(trackedPosition.top())
        t_w = int(trackedPosition.width())
        t_h = int(trackedPosition.height())

        cv2.rectangle(output_image, (t_x, t_y), (t_x + t_w, t_y + t_h), (255, 0, 0), 4)
        carCurrentPosition[carID] = [t_x, t_y, t_w, t_h]

    # Tinh toan frame per second
    end_time = time.time()
    if not (end_time == start_time):
        fps = 1.0 / (end_time - start_time)

    # Lap qua cac xe da track va tinh toan toc do
    for i in carStartPosition.keys():
        [x1, y1, w1, h1] = carStartPosition[i]
        [x2, y2, w2, h2] = carCurrentPosition[i]
        p0 = (int(x2 + (w2 - x2) / 2), int(y2 + (h2 - y2) / 2))
        ct1 = cv2.pointPolygonTest(lane1, p0, False)
        ct2 = cv2.pointPolygonTest(lane2, p0, False)
        ct3 = cv2.pointPolygonTest(lane3, p0, False)
        ct4 = cv2.pointPolygonTest(lane4, p0, False)

        color = (255, 0, 0) if ct1 == 1 else (0, 255, 0) if ct2 == 1 else (255, 0, 255) if ct3 == 1 else (
        0, 255, 255) if ct4 == 1 else (0, 0, 255)
        cv2.rectangle(image, (x2, y2), (w2, h), color, 4)
        carStartPosition[i] = [x2, y2, w2, h2]

        # Neu xe co di chuyen thi
        if [x1, y1, w1, h1] != [x2, y2, w2, h2]:
            # Neu nhu chua tinh toan toc do va toa do hien tai < 200 thi tinh toan toc do
            if (speed[i] is None or speed[i] == 0) and y2 < 200:
                speed[i] = calculate_speed([x1, y1, w1, h1], [x2, y2, w2, h2], fps)

            # Neu nhu da tinh toc do va xe da vuot qua tung do 200 thi hien thi tong do
            if speed[i] is not None and y2 >= 200:
                if speed[i] > 80:
                    cv2.putText(output_image, str(int(speed[i]) - 80) + " km/h vuot qua",
                                (x2, y2), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (0, 0, 255), 2)
                # cv2.putText(output_image, str (int(carID[i]))
                # 		(x2,  y2),cv2.FONT_HERSHEY_SIMPLEX, 1,
                # 		(0, 255, 255), 2)
                if speed[i] < 80:
                    cv2.putText(output_image, str(int(speed[i])) + " km/h",
                                (x2 + w2, y2), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (0, 255, 255), 2)
        # if speed[i] >= 80:
        #   	cv2.putText(output_image, str(int(speed[i])) + " km/h LIMIT",
        #    				(x2, y2), cv2.FONT_HERSHEY_SIMPLEX, 1,
        #    				(0, 0, 255), 2)
    if writer is None:
        # Constructing code of the codec to be used in the function VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        # Writing current processed frame into the video file
        writer = cv2.VideoWriter('result-20km.mp4', fourcc, 30,
                                 (output_image.shape[1], output_image.shape[0]), True)
    # Write processed current frame to the file
    writer.write(output_image)
    cv2.imshow('video', output_image)
    # Detect phim Q
    if cv2.waitKey(1) == ord('q'):
        break

cv2.destroyAllWindows()
