# -*- coding: utf-8 -*-

"""
Objects Detection on Image with YOLO v3 and OpenCV
File: yolov3-video-detection.py
"""

# Detecting Objects on Image with OpenCV deep learning library
#
# How does YOLO-v3 Algorithm works for this example case:
# STEP1: Reading input video
# STEP2: Loading YOLO v3 Network
# STEP3: Reading frames in the loop
# STEP4: Getting blob from the frame
# STEP4: Implementing Forward Pass
# STEP5: Getting Bounding Boxes
# STEP6: Non-maximum Suppression
# STEP7: Drawing Bounding Boxes with Labels
# STEP8: creating a new video by writing processed frames
#
# Result:
# New video file with Detected Objects, Bounding Boxes and Labels


# Importing needed libraries
from flask import Flask, request, make_response
import uuid
import math

# import numpy
import numpy as np
import cv2
import time
import imutils
# from speed_tracking.sort import *

# import pandas

print(cv2.__version__)


app = Flask(__name__)

@app.route('/<path:videoPath>', methods=['POST', 'GET'])
def detection(videoPath):
    print(videoPath)
    app.logger.warning(request.data)
    # Respond with another event (optional)
    response = make_response({
        "msg": "Hi from video-detection-python app! Video path : {}".format(str(videoPath))
    })
    response.headers["Ce-Id"] = str(uuid.uuid4())
    response.headers["Ce-specversion"] = "0.3"
    response.headers["Ce-Source"] = "knative/eventing/samples/hello-world"
    response.headers["Ce-Type"] = "dev.knative.samples.hifromknative"

    pixels_per_meter = 6

    # NOTE:
    # Defining 'VideoCapture' object and reading video from a file make sure that the path and file name is correct
    video = cv2.VideoCapture(videoPath)

    # try to determine the total number of frames in the video file
    try:
        prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
            else cv2.CAP_PROP_FRAME_COUNT
        total = int(video.get(prop))
        print("[INFO] {} total frames in video".format(total))

    except:
        print("[INFO] could not determine # of frames in video")
        print("[INFO] no approx. completion time can be provided")
        total = -1

    writer = None

    h, w = None, None
    with open('yolov3.txt') as f:
        # Getting labels reading every line
        # and putting them into the list
        labels = [line.strip() for line in f]

    network = cv2.dnn.readNetFromDarknet('yolov3.cfg',
                                        'yolov3.weights')

    # Getting list with names of all layers from YOLO v3 network
    layers_names_all = network.getLayerNames()

    layers_names_output = [layers_names_all[i - 1] for i in network.getUnconnectedOutLayers()]
    probability_minimum = 0.5
    threshold = 0.3

    colours = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')


    # Defining variable for counting frames at the end we will show total amount of processed frames
    f = 0

    # Defining variable for counting total time At the end we will show time spent for processing all frames
    t = 0
    # Data export
    data_export = []
    while True:
    # Capturing frame-by-frame
        ret, frame = video.read()

        # If the frame was not retrieved e.g.: at the end of the video, then we break the loop
        if not ret:
            break

        # Getting spatial dimensions of the frame as we do it only once from the very beginning
        # all other frames have the same dimension
        if w is None or h is None:
            # Slicing from tuple only first two elements
            h, w = frame.shape[:2]

        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
                                    swapRB=True, crop=False)

        network.setInput(blob)  # setting blob as input to the network
        start = time.time()
        output_from_network = network.forward(layers_names_output)
        end = time.time()

        # Increasing counters for frames and total time
        f += 1
        t += end - start

        # Showing spent time for single current frame
        print('Frame number {0} took {1:.5f} seconds'.format(f, end - start))

        bounding_boxes = []
        confidences = []
        classIDs = []
        carPrePosition = []
        carCurrentPosition = []

        # Going through all output layers after feed forward pass
        for result in output_from_network:
            # Going through all detections from current output layer
            for detected_objects in result:
                # Getting 80 classes' probabilities for current detected object
                scores = detected_objects[5:]
                # Getting index of the class with the maximum value of probability
                class_current = np.argmax(scores)
                # Getting value of probability for defined class
                confidence_current = scores[class_current]

                if confidence_current > probability_minimum:
                    box_current = detected_objects[0:4] * np.array([w, h, w, h])

                    # Now, from YOLO data format, we can get top left corner coordinates that are x_min and y_min
                    x_center, y_center, box_width, box_height = box_current
                    x_min = int(x_center - (box_width / 2))
                    y_min = int(y_center - (box_height / 2))

                    # Adding results into prepared lists
                    bounding_boxes.append([x_min, y_min, int(box_width), int(box_height)])
                    confidences.append(float(confidence_current))
                    classIDs.append(class_current)

            # print("boxes_length", len(bounding_boxes))

        results = cv2.dnn.NMSBoxes(bounding_boxes, confidences,
                                probability_minimum, threshold)

        if len(results) > 0:
            # Going through indexes of results
            for i in results.flatten():
                # Getting current bounding box coordinates, its width and height
                x_min, y_min = bounding_boxes[i][0], bounding_boxes[i][1]
                box_width, box_height = bounding_boxes[i][2], bounding_boxes[i][3]

                # Preparing colour for current bounding box and converting from numpy array to list
                colour_box_current = colours[classIDs[i]].tolist()

                # print(type(colour_box_current))  # <class 'list'>
                # print(colour_box_current)  # [172 , 10, 127]

                # Drawing bounding box on the original current frame
                cv2.rectangle(frame, (x_min, y_min),
                            (x_min + box_width, y_min + box_height),
                            colour_box_current, 2)

                # calculate speed

                # [x1, y1, h1, w1] = carPrePosition[i]
                # [x2, y2, h2, w2] = bounding_boxes[i]
                #
                # speed_current = calculate_speed([x1, y1, h1, w1], [x2, y2, h2, w2], 30)

                # Preparing text with label and confidence for current bounding box
                text_box_current = '{}: {:.4f}'.format(labels[int(classIDs[i])],
                                                    confidences[i])

                data_export.append([f, i, labels[int(classIDs[i])], confidences[i], bounding_boxes[i],

                                    len(bounding_boxes)])


                cv2.putText(frame, text_box_current, (x_min, y_min - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour_box_current, 2)
            print(data_export)


        # Initializing writer
        # we do it only once from the very beginning when we get spatial dimensions of the frames
        if writer is None:
            # Constructing code of the codec to be used in the function VideoWriter
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')

            # Writing current processed frame into the video file
            writer = cv2.VideoWriter('result-detect-20km.mp4', fourcc, 30,
                                    (frame.shape[1], frame.shape[0]), True)

        # Write processed current frame to the file
        # writer.write(frame)

        # cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

# Printing final results
    print()
    print('Total number of frames', f)
    print('Total amount of time {:.5f} seconds'.format(t))
    # print('FPS:', round((f / t), 1))

    # Releasing video reader and writer
    video.release()
    writer.release()

    return response

# Save data to csv
# data_frame = pandas.DataFrame(data_export)
# data_frame.to_csv("20km.csv")
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)
