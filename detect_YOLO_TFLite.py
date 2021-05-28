#! /usr/bin/env python


import numpy as np
import tensorflow as tf
from time import time
import json

import logging as log
log.basicConfig(format='[%(levelname)s] %(message)s', level=log.INFO)

IMG_FILE = "/dataset/RZSS_images/1_animal_empty_r/animal/PICT0006.JPG"
FOLDER = "/dataset/RZSS_images/1_animal_empty_r/animal/"
NIMGS = 20

MODEL_PATH = "/repos/training_DL/frameworks_/raccoon.tflite"
MODEL_PATH = "/repos/training_DL/frameworks_/in_out_models/output_model.tflite"

OUTPUT_PATH = "./output_tflite/"

CONFIG_PATH = "./models/config_w_m.json"

net_h, net_w = 416, 416 # a multiple of 32, the smaller the faster
obj_thresh, nms_thresh = 0.5, 0.45

## config
with open(CONFIG_PATH) as config_buffer:    
    config = json.load(config_buffer)

anchors = config['model']['anchors']#anchors = [72,120, 77,71, 78,215, 110,90, 120,267, 126,143, 190,139, 200,273, 348,350]
labels = config['model']['labels']

import os
if FOLDER:
    files = [os.path.join(FOLDER,f) for f in os.listdir(FOLDER)]
    files.sort()
else:
    files = [IMG_FILE]

# -------------------------------------------------------------------
# functions from https://github.com/experiencor/keras-yolo3
# -------------------------------------------------------------------
def preprocess_input(image, net_h, net_w):
    new_h, new_w, _ = image.shape

    # determine the new size of the image
    if (float(net_w)/new_w) < (float(net_h)/new_h):
        new_h = (new_h * net_w)//new_w
        new_w = net_w
    else:
        new_w = (new_w * net_h)//new_h
        new_h = net_h

    # resize the image to the new size
    resized = cv2.resize(image[:,:,::-1]/255., (new_w, new_h))

    # embed the image into the standard letter box
    new_image = np.ones((net_h, net_w, 3)) * 0.5
    new_image[(net_h-new_h)//2:(net_h+new_h)//2, (net_w-new_w)//2:(net_w+new_w)//2, :] = resized
    new_image = np.expand_dims(new_image, 0)

    return new_image

def decode_netout(netout, anchors, obj_thresh, net_h, net_w):
    grid_h, grid_w = netout.shape[:2]
    nb_box = 3
    netout = netout.reshape((grid_h, grid_w, nb_box, -1))
    nb_class = netout.shape[-1] - 5

    boxes = []

    netout[..., :2]  = _sigmoid(netout[..., :2])
    netout[..., 4]   = _sigmoid(netout[..., 4])
    netout[..., 5:]  = netout[..., 4][..., np.newaxis] * _softmax(netout[..., 5:])
    netout[..., 5:] *= netout[..., 5:] > obj_thresh

    for i in range(grid_h*grid_w):
        row = i // grid_w
        col = i % grid_w
        
        for b in range(nb_box):
            # 4th element is objectness score
            objectness = netout[row, col, b, 4]
            
            if(objectness <= obj_thresh): continue
            
            # first 4 elements are x, y, w, and h
            x, y, w, h = netout[row,col,b,:4]

            x = (col + x) / grid_w # center position, unit: image width
            y = (row + y) / grid_h # center position, unit: image height
            w = anchors[2 * b + 0] * np.exp(w) / net_w # unit: image width
            h = anchors[2 * b + 1] * np.exp(h) / net_h # unit: image height  
            
            # last elements are class probabilities
            classes = netout[row,col,b,5:]
            
            box = BoundBox(x-w/2, y-h/2, x+w/2, y+h/2, objectness, classes)

            boxes.append(box)

    return boxes

def correct_yolo_boxes(boxes, image_h, image_w, net_h, net_w):
    if (float(net_w)/image_w) < (float(net_h)/image_h):
        new_w = net_w
        new_h = (image_h*net_w)/image_w
    else:
        new_h = net_w
        new_w = (image_w*net_h)/image_h
        
    for i in range(len(boxes)):
        x_offset, x_scale = (net_w - new_w)/2./net_w, float(new_w)/net_w
        y_offset, y_scale = (net_h - new_h)/2./net_h, float(new_h)/net_h
        
        boxes[i].xmin = int((boxes[i].xmin - x_offset) / x_scale * image_w)
        boxes[i].xmax = int((boxes[i].xmax - x_offset) / x_scale * image_w)
        boxes[i].ymin = int((boxes[i].ymin - y_offset) / y_scale * image_h)
        boxes[i].ymax = int((boxes[i].ymax - y_offset) / y_scale * image_h)

def _interval_overlap(interval_a, interval_b):
    x1, x2 = interval_a
    x3, x4 = interval_b

    if x3 < x1:
        if x4 < x1:
            return 0
        else:
            return min(x2,x4) - x1
    else:
        if x2 < x3:
             return 0
        else:
            return min(x2,x4) - x3 

def bbox_iou(box1, box2):
    intersect_w = _interval_overlap([box1.xmin, box1.xmax], [box2.xmin, box2.xmax])
    intersect_h = _interval_overlap([box1.ymin, box1.ymax], [box2.ymin, box2.ymax])  
    
    intersect = intersect_w * intersect_h

    w1, h1 = box1.xmax-box1.xmin, box1.ymax-box1.ymin
    w2, h2 = box2.xmax-box2.xmin, box2.ymax-box2.ymin
    
    union = w1*h1 + w2*h2 - intersect
    
    return float(intersect) / union

def do_nms(boxes, nms_thresh):
    if len(boxes) > 0:
        nb_class = len(boxes[0].classes)
    else:
        return
        
    for c in range(nb_class):
        sorted_indices = np.argsort([-box.classes[c] for box in boxes])

        for i in range(len(sorted_indices)):
            index_i = sorted_indices[i]

            if boxes[index_i].classes[c] == 0: continue

            for j in range(i+1, len(sorted_indices)):
                index_j = sorted_indices[j]

                if bbox_iou(boxes[index_i], boxes[index_j]) >= nms_thresh:
                    boxes[index_j].classes[c] = 0

def _sigmoid(x):
    from scipy.special import expit
    return expit(x)
def _softmax(x, axis=-1):
    x = x - np.amax(x, axis, keepdims=True)
    e_x = np.exp(x)  
    return e_x / e_x.sum(axis, keepdims=True)

class BoundBox:
    def __init__(self, xmin, ymin, xmax, ymax, c = None, classes = None):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        
        self.c       = c
        self.classes = classes

        self.label = -1
        self.score = -1

    def get_label(self):
        if self.label == -1:
            self.label = np.argmax(self.classes)
        
        return self.label
    
    def get_score(self):
        if self.score == -1:
            self.score = self.classes[self.get_label()]
            
        return self.score     

def decode_netout(netout, anchors, obj_thresh, net_h, net_w):
    grid_h, grid_w = netout.shape[:2]
    nb_box = 3
    netout = netout.reshape((grid_h, grid_w, nb_box, -1))
    nb_class = netout.shape[-1] - 5

    boxes = []

    netout[..., :2]  = _sigmoid(netout[..., :2])
    netout[..., 4]   = _sigmoid(netout[..., 4])
    netout[..., 5:]  = netout[..., 4][..., np.newaxis] * _softmax(netout[..., 5:])
    netout[..., 5:] *= netout[..., 5:] > obj_thresh

    for i in range(grid_h*grid_w):
        row = i // grid_w
        col = i % grid_w
        
        for b in range(nb_box):
            # 4th element is objectness score
            objectness = netout[row, col, b, 4]

            if(objectness <= obj_thresh): continue
            
            # first 4 elements are x, y, w, and h
            x, y, w, h = netout[row,col,b,:4]

            x = (col + x) / grid_w # center position, unit: image width
            y = (row + y) / grid_h # center position, unit: image height
            w = anchors[2 * b + 0] * np.exp(w) / net_w # unit: image width
            h = anchors[2 * b + 1] * np.exp(h) / net_h # unit: image height  
            
            # last elements are class probabilities
            classes = netout[row,col,b,5:]
            
            box = BoundBox(x-w/2, y-h/2, x+w/2, y+h/2, objectness, classes)

            boxes.append(box)

    return boxes






# Load the TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print(output_details)
# Test the model on random input data.
input_shape = input_details[0]['shape']
#input_shape[0] = 8; # batch size
input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
interpreter.set_tensor(input_details[0]['index'], input_data)

print(input_shape )

## real input image
#H = input_shape[1]; W = input_shape[2];
import cv2


batch_boxes  = []#[None]*len(files[:NIMGS])
images = []

for IMG in files[:NIMGS]:
    print('------------------------')
    print(IMG)
    img = cv2.imread(IMG, cv2.IMREAD_COLOR) #
    image_h, image_w, _ = img.shape
    images.append( img )
    print( img.shape)
    img = preprocess_input(img, net_h, net_w).astype(np.float32)

    interpreter.set_tensor(input_details[0]['index'], img )

    Nruns = 1#10
    t0 = time()
    for k in range(Nruns):
        interpreter.invoke()
    t1 = time()
    print( 'Time : ' + str( 1000*(t1 - t0)/Nruns ))
    
    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
    output_data=[]
    for j in range(3):
        output_data.append( interpreter.get_tensor(output_details[j]['index']).squeeze() )
        print(output_data[-1].shape)

    yolos = output_data
    boxes = []
    # decode the output of the network
    for j in range(len(yolos)):
        yolo_anchors = anchors[(2-j)*6:(3-j)*6] # config['model']['anchors']
        boxes += decode_netout(yolos[j], yolo_anchors, obj_thresh, net_h, net_w)
        #print(yolos[j], yolo_anchors, obj_thresh, net_h, net_w)

    # correct the sizes of the bounding boxes
    try:
        correct_yolo_boxes(boxes, image_h, image_w, net_h, net_w)
    except:
        print('WARNING: exception in correct_yolo_boxes()')

    # suppress non-maximal boxes
    do_nms(boxes, nms_thresh)        
    batch_boxes.append(  boxes )
    #print(boxes); import sys; sys.exit(0)
    if len(boxes) > 0:
        pboxes = np.array([[box.xmin, box.ymin, box.xmax, box.ymax, box.get_score()] for box in boxes])
    print(pboxes)
    #print(boxes[0].xmin, boxes[0].ymin, boxes[0].xmax, boxes[0].ymax, boxes[0].c, boxes[0].classes ); import sys; sys.exit(0)


    
def draw_boxes(image, boxes, labels, obj_thresh, quiet=True):
    for box in boxes:
        label_str = ''
        label = -1
        
        for i in range(len(labels)):
            if box.classes[i] > obj_thresh:
                if label_str != '': label_str += ', '
                label_str += (labels[i] + ' ' + str(round(box.get_score()*100, 2)) + '%')
                label = i
            if not quiet: print(label_str)

                
        if label >= 0:
            text_size = cv2.getTextSize(label_str, cv2.FONT_HERSHEY_SIMPLEX, 1.1e-3 * image.shape[0], 5)
            width, height = text_size[0][0], text_size[0][1]
            region = np.array([[box.xmin-3,        box.ymin], 
                               [box.xmin-3,        box.ymin-height-26], 
                               [box.xmin+width+13, box.ymin-height-26], 
                               [box.xmin+width+13, box.ymin]], dtype='int32')  

            cv2.rectangle(img=image, pt1=(box.xmin,box.ymin), pt2=(box.xmax,box.ymax), color=[0,0,255], thickness=5)
            cv2.fillPoly(img=image, pts=[region], color=[0,0,255])
            cv2.putText(img=image, 
                        text=label_str, 
                        org=(box.xmin+13, box.ymin - 13), 
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                        fontScale=1e-3 * image.shape[0], 
                        color=(0,0,0), 
                        thickness=2)
        
    return image


for i in range(len(files[:NIMGS])):
    print(images[i].shape)
    bbox0 = batch_boxes[i]
    #bbox0 = [batch_boxes[i][0]] if len(batch_boxes[i]) else []
    #if bbox0==[]: continue
    #boxes = bbox0;print(boxes[0].xmin, boxes[0].ymin, boxes[0].xmax, boxes[0].ymax, boxes[0].c, boxes[0].classes ); #import sys; sys.exit(0)

    if bbox0:
        draw_boxes(images[i], bbox0, labels, obj_thresh) 
        #cv2.imshow('video with bboxes', images[i])
        #if cv2.waitKey(1) == 27: 
        #    break  # esc to quit

        # write the image with bounding boxes to file
        cv2.imwrite(OUTPUT_PATH+'{}.JPG'.format(i), np.uint8(images[i]))    
        print('OUTPUT SAVED AS ' +OUTPUT_PATH+'{}.JPG'.format(i))  

#cv2.destroyAllWindows()        


    
print( input_details[0])  