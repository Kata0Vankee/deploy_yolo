import cv2
import numpy as np
import pathlib


# Load YOLO
def load_model():
	path = pathlib.Path(__file__).parent
	print(path)
	print((path/'yolov3.weights').exists())
	print((path/'yolov3.cfg').exists())
	
	try:
		net = cv2.dnn.readNet("src/yolov3.weights", "src/yolov3.cfg") 
	except Exception as e:
		print(e)

	classes = []

	with open("src/coco.names", "r") as f:
		classes = [line.strip() for line in f.readlines()]

	layer_names = net.getLayerNames()
	#print(layer_names)
	output_layers = [layer_names[layer[0] - 1] for layer in net.getUnconnectedOutLayers()]
	return (net, classes, output_layers)


def detect_img():
	net, classes, output_layers = load_model()
	# img_path = 'img/' + img_name + '.jpg'
	# print(path/img_path)

	# print("path::: "+img_path)
	img = cv2.imread('src/img/pic1.jpg')
	img = cv2.resize(img, None, fx=0.4, fy=0.4)
	height, width, channels = img.shape

	#Decting object
	blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0,0,0), True, crop=False)
	net.setInput(blob)
	outs = net.forward(output_layers)

	# for b in blob:
	#     for n, img_blob in enumerate(b):
	#         cv2.imshow(str(n), img_blob)

	class_ids = []
	confidences = []
	boxes = []
	for out in outs:
	    for detection in out:
	        scores = detection[5:]   #Because detection[0:5] are the info of the boxes
	        class_id = np.argmax(scores)
	        confidence = scores[class_id]
	        if confidence > 0.5:
	            #Object detected
	            center_x = int(detection[0] * width)
	            center_y = int(detection[1] * height)
	            w = int(detection[2] * width)
	            h = int(detection[3] * height)

	            #Rectangle coordinates
	            x = int(center_x - w/2)
	            y = int(center_y - h/2)

	            boxes.append([x, y, w, h])
	            confidences.append(float(confidence))
	            class_ids.append(class_id)

	indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
	print(indexes)

	font = cv2.FONT_HERSHEY_COMPLEX

	for index in range(len(boxes)):
	    if index in indexes:
	        x, y, width, height = boxes[index]
	        confidence = confidences[index]
	        label = str(classes[class_ids[index]]) + " : " + str(round(confidence, 4))
	        color = (0, 255, 0)
	        cv2.rectangle(img, (x, y), (x + width, x + height), color, 1)
	        cv2.putText(img, label, (x, y + 10), font, 0.5, color, 1)

	cv2.imwrite('src/img/pic1_detected.jpg', img)
	return
