import os
import sys
import numpy as np
from PIL import Image
# import requests
from bs4 import BeautifulSoup
from numpy import expand_dims
import tensorflow as tf
from skimage.transform import resize
from keras.models import load_model, Model
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from time import time
import flask
import io
from flask import Flask, render_template, request
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

#Load chrome drive
CHROME_PATH = 'chromedriver.exe' 
options = Options()  
options.add_argument("--headless") # This make Chrome run headless

app = flask.Flask(__name__)
yolov3 = None

labels = []
with open('logos.name','r+') as f:
    for i in f:
        i = i.rstrip()
        labels.append(i)

#Bank dict map to real domain
nametoL = {'ABBANK': 'abbank.vn',
           'ACB': 'acb.com.vn',
           'AGRIBANK': 'agribank.com.vn',
           'BACA-BANK': 'baca-bank.vn',
           'BAOVIETBANK': 'baovietbank.vn',
           'BIDC': 'bidc.vn', 'BIDV': 'bidv.com.vn',
           'CITIBANK': 'citibank.com.vn', 'DB': 'db.com',
           'DONGABANK': 'dongabank.com.vn',
           'EXIMBANK': 'eximbank.com.vn',
           'GPBANK': 'gpbank.com.vn',
           'HDBANK': 'hdbank.com.vn',
           'HLB': 'hlb.com.my',
           'INDOVINABANK': 'indovinabank.com.vn',
           'KIENLONGBANK': 'kienlongbank.com',
           'LIENVIETPOSTBANK': 'lienvietpostbank.com.vn',
           'MBBANK': 'mbbank.com.vn',
           'MSB': 'msb.com.vn',
           'NAMABANK': 'namabank.com.vn',
           'NCB-BANK': 'ncb-bank.vn',
           'OCB': 'ocb.com.vn',
           'OCEANBANK': 'oceanbank.vn',
           'PGBANK': 'pgbank.com.vn',
           'PUBLICBANK': 'publicbank.com.vn',
           'PVCOMBANK': 'pvcombank.com.vn',
           'SACOMBANK': 'sacombank.com.vn',
           'SAIGONBANK': 'saigonbank.com.vn',
           'SCB': 'scb.com.vn',
           'SEABANK': 'seabank.com.vn',
           'SHB': 'shb.com.vn',
           'SHINHAN': 'shinhan.com.vn',
           'STANDARDCHARTERED': 'standardchartered.com',
           'TECHCOMBANK': 'techcombank.com.vn',
            'TPB': 'tpb.vn',
            'VBSP': 'vbsp.org.vn',
            'VIB': 'vib.com.vn',
            'VIETABANK': 'vietabank.com.vn',
           'VIETBANK': 'vietbank.com.vn',
            'VIETCAPITALBANK': 'vietcapitalbank.com.vn',
            'VIETCOMBANK': 'vietcombank.com.vn',
            'VIETINBANK': 'vietinbank.vn',
            'VPBANK': 'vpbank.com.vn',
            'VRBANK': 'vrbank.com.vn',
            'CBBANK' : '123'}

valid_ext = ['rgb','gif','pbm','pgm','ppm','tiff','rast','xbm','jpeg', 'jpg','bmp','png','webp','exr']
graph = tf.compat.v1.get_default_graph()

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

def load():
    global yolov3
    global graph
    graph = tf.get_default_graph()
    yolov3 = load_model('30k.h5',compile=False)

#Preprocessing Image
def load_image_pixels(filename, shape):
    image = load_img(filename)
    width, height = image.size
    image = load_img(filename, target_size=shape)
    image = img_to_array(image)
    image = image.astype('float32')
    image /= 255.0
    image = expand_dims(image, 0)
    return image, width, height

class BoundBox:
    def __init__(self, xmin, ymin, xmax, ymax, objness = None, classes = None):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        
        self.objness = objness
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

def _sigmoid(x):
    return 1. / (1. + np.exp(-x))

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

def decode_netout(netout, anchors, obj_thresh,  net_h, net_w):
    grid_h, grid_w = netout.shape[:2]
    nb_box = 3
    netout = netout.reshape((grid_h, grid_w, nb_box, -1))
    nb_class = netout.shape[-1] - 5

    boxes = []

    netout[..., :2]  = _sigmoid(netout[..., :2])
    netout[..., 4:]  = _sigmoid(netout[..., 4:])
    netout[..., 5:]  = netout[..., 4][..., np.newaxis] * netout[..., 5:]
    netout[..., 5:] *= netout[..., 5:] > obj_thresh

    for i in range(grid_h*grid_w):
        row = i / grid_w
        col = i % grid_w
        
        for b in range(nb_box):
            objectness = netout[int(row)][int(col)][b][4]
            
            if(objectness.all() <= obj_thresh): continue
            
            x, y, w, h = netout[int(row)][int(col)][b][:4]

            x = (col + x) / grid_w 
            y = (row + y) / grid_h 
            w = anchors[2 * b + 0] * np.exp(w) / net_w 
            h = anchors[2 * b + 1] * np.exp(h) / net_h  
            
            classes = netout[int(row)][col][b][5:]
            
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

def get_boxes(boxes, labels, thresh):
    v_boxes, v_labels, v_scores = list(), list(), list()
    for box in boxes:
        for i in range(len(labels)):
            if box.classes[i] > thresh:
                v_boxes.append(box)
                v_labels.append(labels[i])
                v_scores.append(box.classes[i]*100)
    return v_boxes, v_labels, v_scores

#under maintain (check website has password form)
def check_form(link):
    result = requests.get(link)
    soup = BeautifulSoup(result.content, "html.parser")
    inputs = soup.find_all('input',{'type' : 'password'})
    if inputs != None:
        return 1
    else:
        return 0
    # return 1
#Index
# @app.route("/index")
# def main():
#    return render_template("index.html")

@app.route("/index",methods = ['POST', 'GET'])
def predict():
    return render_template("index.html")
    net_h, net_w = 416, 416
    obj_thresh, nms_thresh = 0.5, 0.45
    anchors = [[116,90,  156,198,  373,326],  [30,61, 62,45,  59,119], [10,13,  16,30,  33,23]]
    input_w, input_h = 416, 416

    data = {"success": False}

    if flask.request.method == "POST":
            result = request.form
            link = result['link']

            #Take screenshot with selenium
            driver = webdriver.Chrome(executable_path=CHROME_PATH, options = options ) 
            driver.get(link)
            name = link.split("/")
            filename = name[2]+'.png'
            driver.save_screenshot( filename )
            driver.close()

            #Load Image from server
            photo_filename = filename
            img = Image.open(photo_filename)
            image, image_w, image_h = load_image_pixels(photo_filename, (input_w, input_h))

            class_threshold = 0.6
            with graph.as_default():
                yolos = yolov3.predict(image)
                
                data["predictions"] = []

                boxes = list()

                beauty_alert = []

                for i in range(len(yolos)):
                    boxes += decode_netout(yolos[i][0], anchors[i], obj_thresh,  net_h, net_w)

                v_boxes, v_labels, v_scores = get_boxes(boxes, labels, class_threshold)
                
                v_labels = list(dict.fromkeys(v_labels))
                if len(v_labels) == 1 and link != nametoL[v_labels[0]]:
                    icon = 'warning'
                    # print('THAG L* nay gia mao website '+nametoL[v_labels[0]]) 
                    beauty_alert.append('This web is not safe')
                    beauty_alert.append('The logo belong to domain '+nametoL[v_labels[0]])
                    beauty_alert.append(icon)
                    return render_template("index.html",beauty_alert=beauty_alert)
                if len(v_labels) == 0 or len(v_labels) > 1 :
                    icon = 'success'
                    beauty_alert.append('This web is Safe')
                    beauty_alert.append('test')
                    beauty_alert.append(icon)
                    return render_template("index.html",beauty_alert=beauty_alert)
                # r =  dict((x,v_labels.count(x)) for x in set(v_labels))
                # r['website'] = link
                # r['password-form'] = check_form(link)
                # data["predictions"].append(r)
            # data["success"] = True


    #In case want to respone in json type uncomment this
    # return flask.jsonify(data)
if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
        "please wait until server has fully started"))
    load()
    app.run(debug=True)