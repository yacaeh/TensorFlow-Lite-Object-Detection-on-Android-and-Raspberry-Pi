#!/tflite1-env1
# Import packages
import os
import argparse
import cv2
import numpy as np
import sys
import time
from threading import Thread
import importlib.util
import csv
import threading
from datetime import datetime
import logging
from logging.handlers import RotatingFileHandler
import logging.config
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
from TimerReset import TimerReset
import ssl
from urllib.parse import urlparse
import json
import cgi

# import RPi.GPIO as GPIO

# GPIO_PWM=12
# GPIO.setmode(GPIO.BCM)
# GPIO.setup(GPIO_PWM, GPIO.OUT)
# pwm_led=GPIO.PWM(GPIO_PWM, 100)
# pwm_led.start(0)
# GPIO.setwarnings(False)


# Light sensor related variables
#import smbus # i2c library
import time
# i2c channel number
I2C_CH=1

# BH1750 address
BH1750_DEV_ADDR=0x23


CONT_H_RES_MODE = 0X10
CONT_H_RES_MODE2 = 0X11
CONT_L_RES_MODE = 0X13
ONETIME_H_RES_MODE = 0X20
ONETIME_H_RES_MODE2 = 0X21
ONETIME_L_RES_MODE = 0X23

# create i2c channel library
#i2c=smbus.SMBus(I2C_CH)
# read 2BYTE as CONT_HRES_MODE
time.sleep(1)



# Define and parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--modeldir', help='Folder the .tflite file is located in',
                    required=True)
parser.add_argument('--graph', help='Name of the .tflite file, if different than detect.tflite',
                    default='detect.tflite')
parser.add_argument('--labels', help='Name of the labelmap file, if different than labelmap.txt',
                    default='labelmap.txt')
parser.add_argument('--threshold', help='Minimum confidence threshold for displaying detected objects',
                    default=0.5)
parser.add_argument('--resolution', help='Desired webcam resolution in WxH. If the webcam does not support the resolution entered, errors may occur.',
                    default='1280x720')
parser.add_argument('--edgetpu', help='Use Coral Edge TPU Accelerator to speed up detection',
                    action='store_true')

args = parser.parse_args()

MODEL_NAME = args.modeldir
GRAPH_NAME = args.graph
LABELMAP_NAME = args.labels
min_conf_threshold = float(args.threshold)
resW, resH = args.resolution.split('x')
imW, imH = int(resW), int(resH)
use_TPU = args.edgetpu
capture = None
current_lightData = None
isPerson = False
personFound = False
isTimer = False
# Log conifgs
current_dir = os.path.dirname(os.path.realpath(__file__))
current_file = os.path.basename(__file__)
current_file_name = current_file[:-3]  # xxxx.py
LOG_FILENAME = 'log-{}'.format(current_file_name)

log_dir = '{}/logs'.format(current_dir)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# 로거 생성
myLogger = logging.getLogger('test') # 로거 이름: test
myLogger.setLevel(logging.INFO) # 로깅 수준: INFO

# 핸들러 생성
file_handler = logging.handlers.TimedRotatingFileHandler(
  filename=log_dir+'/'+LOG_FILENAME, when='midnight', interval=1,  encoding='utf-8'
  ) # 자정마다 한 번씩 로테이션
file_handler.suffix = 'log-%Y%m%d' # 로그 파일명 날짜 기록 부분 포맷 지정 

myLogger.addHandler(file_handler) # 로거에 핸들러 추가
formatter = logging.Formatter(
  '%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] %(message)s'
  )
file_handler.setFormatter(formatter) # 핸들러에 로깅 포맷 할당

# Import TensorFlow libraries
# If tflite_runtime is installed, import interpreter from tflite_runtime, else import from regular tensorflow
# If using Coral Edge TPU, import the load_delegate library
pkg = importlib.util.find_spec('tflite_runtime')
if pkg:
    from tflite_runtime.interpreter import Interpreter
    if use_TPU:
        from tflite_runtime.interpreter import load_delegate
else:
    from tensorflow.lite.python.interpreter import Interpreter
    if use_TPU:
        from tensorflow.lite.python.interpreter import load_delegate

# If using Edge TPU, assign filename for Edge TPU model
if use_TPU:
    # If user has specified the name of the .tflite file, use that name, otherwise use default 'edgetpu.tflite'
    if (GRAPH_NAME == 'detect.tflite'):
        GRAPH_NAME = 'edgetpu.tflite'       

# Get path to current working directory
CWD_PATH = os.getcwd()

# Path to .tflite file, which contains the model that is used for object detection
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,GRAPH_NAME)

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,MODEL_NAME,LABELMAP_NAME)

# Load the label map
with open(PATH_TO_LABELS, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Have to do a weird fix for label map if using the COCO "starter model" from
# https://www.tensorflow.org/lite/models/object_detection/overview
# First label is '???', which has to be removed.
if labels[0] == '???':
    del(labels[0])

# Load the Tensorflow Lite model.
# If using Edge TPU, use special load_delegate argument
if use_TPU:
    interpreter = Interpreter(model_path=PATH_TO_CKPT,
                            experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
    print(PATH_TO_CKPT)
else:
    interpreter = Interpreter(model_path=PATH_TO_CKPT)

interpreter.allocate_tensors()

# Get model details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

floating_model = (input_details[0]['dtype'] == np.float32)

input_mean = 127.5
input_std = 127.5

# Initialize frame rate calculation
frame_rate_calc = 1
freq = cv2.getTickFrequency()

# Reset Timer Value
RESET_SEC = 5


class LightSensor(threading.Thread):
    global lightSensorValue
    global current_lightData
    global exceedFound
    global isSensorTimer
    global RESET_SEC

    sensorTimer = None
    isSensorTimer = False
    exceedFound = False

    def __init__(self):
        self._start = None
        threading.Thread.__init__(self)

    def stop(self):
        self.__stop = True

    def run(self):
        while True:
            try:
                # luxBytes = i2c.read_i2c_block_data(BH1750_DEV_ADDR, CONT_H_RES_MODE, 2)
            # type conversion BYTE to INT
                # lux=int.from_bytes(luxBytes, byteorder='big')
                lux = lux*10
                print('{0} lux'.format(lux))
                lightSensorValue = lux

                isLihgtExceed = False
                isSensorTimer = False
                if lux > 2000:
                    isLightExceed = True
                else:
                    isLihgtExceed = False

                exceedFound = False
                if isLihgtExceed:
                    if exceedFound:
                        if isSensorTimer:
                            print(f"exceed again before timer passed reset {RESET_SEC} sec")
                            sensorTimer.reset(RESET_SEC)
                        else:
                            print("exceedagain")
                
                    else:
                        print("First exceed")
                        setLight(100)
                        exceedFound = True
                
                else:
                    if not exceedFound:
                        print("justwaiting light...")
                    else :
                        if not isSensorTimer:
                            isSensorTimer = True
                            sensorTimer = TimerReset(RESET_SEC, applySensorLightData, args=[eval(current_lightData)], kwargs={})
                            sensorTimer.start()
                            print(f"person not found! wait {RESET_SEC} sec")
                        else:
                            print("waiting for timer to finish...")



                time.sleep(1)
            except IOError:
                print("IOERROR!")
                pass


# Define VideoStream class to handle streaming of video from webcam in separate processing thread
# Source - Adrian Rosebrock, PyImageSearch: https://www.pyimagesearch.com/2015/12/28/increasing-raspberry-pi-fps-with-python-and-opencv/
class VideoStream:
    """Camera object that controls video streaming from the Picamera"""
    def __init__(self,resolution=(640,480),framerate=2,video=0):
        # Initialize the PiCamera and the camera image stream
        self.stream = cv2.VideoCapture(video)
        ret = self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        ret = self.stream.set(3,resolution[0])
        ret = self.stream.set(4,resolution[1])
            
        # Read first frame from the stream
        (self.grabbed, self.frame) = self.stream.read()

	# Variable to control when the camera is stopped
        self.stopped = False

    def start(self):
	# Start the thread that reads frames from the video stream
        Thread(target=self.update,args=()).start()
        return self

    def update(self):
        # Keep looping indefinitely until the thread is stopped
        while True:
            # If the camera is stopped, stop the thread
            if self.stopped:
                # Close camera resources
                self.stream.release()
                return

            # Otherwise, grab the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
	# Return the most recent frame
        return self.frame

    def stop(self):
	# Indicate that the camera and thread should be stopped
        self.stopped = True


def setLight(value):
    #global pwm_led
    #pwm_led.ChangeDutyCycle(value)
    print("SET Light with value",value)

def applySensorLightData(data):
    print(data[2],data[3],data[4])
    global isLightExceed
    global exceedFound
    global isLightTimer

    if isLightExceed:
        lightValue = data[3]
    else :
        lightValue = data[4]
        isLightTimer = False
        exceedFound = False
        
    # lightValue = data[3] if isPerson else data[4]
    print(f"Applying Light Data -isLightExceed: {str(isLightExceed)} Changed light to:{str(data[4])}")
    myLogger.info(f"Applying Light Data -isLightExceed: {str(isLightExceed)} Changed light to:{str(data[4])}")
    setLight(lightValue)

def applyLightData(data):
    print(data[2],data[3],data[4])
    global isPerson
    global personFound
    global isTimer

    if isPerson:
        lightValue = data[3]
    else :
        lightValue = data[4]
        isTimer = False
        personFound = False
        
    # lightValue = data[3] if isPerson else data[4]
    print(f"Applying Light Data -isPerson: {str(isPerson)} Changed light to:{str(data[4])}")
    myLogger.info(f"Applying Light Data -isPerson: {str(isPerson)} Changed light to:{str(data[4])}")
    setLight(lightValue)

class checkingDate(threading.Thread):
    def __init__(self):
        self._start = None
        threading.Thread.__init__(self)

    def run(self):
        self._start = time.time()
        global current_lightData

        while True:
            currentTime = datetime.now()
            month = currentTime.strftime("%m")
            hour = currentTime.strftime("%H")
            
            print('현재 월, 시간:' + month, hour)
            line_to_read = (int(month)-1)*24+int(hour)
            csvData = open('lightData.csv', 'r')
            lines = csvData.readlines()

            current_lightData = lines[line_to_read+1]
            applyLightData(eval(current_lightData))
            # should call applyData

            sys.stdout.flush()
            time.sleep(3600)

class waitingSec(threading.Thread):
    def __init__(self,waitfor=5):
        self._start = None
        threading.Thread.__init__(self)

    def stop(self):
        self.__stop = True

    def run(self):
        while True:
            time.sleep(5)

frame = None
c = checkingDate()
c.start()
isVideo = False
class initDetector(threading.Thread):
    
    def __init__(self):
        threading.Thread.__init__(self)

    def run(self):
        global isVideo
        global MODEL_NAME
        global GRAPH_NAME
        global LABELMAP_NAME
        global min_conf_threshold
        global resW, resH
        global imW, imH
        global use_TPU
        global capture
        global current_lightData
        global isPerson 
        global frame_rate_calc
        global frame
        global width, height
        global current_lightData
        global personFound
        global isTimer
        global RESET_SEC

        videostream = VideoStream(resolution=(imW,imH),framerate=8,video=0).start()
        time.sleep(1)

        timer = None
        isTimer = False
        personFound = False

        #for frame1 in camera.capture_continuous(rawCapture, format="bgr",use_video_port=True):
        while True:
            print("detecting!")
            # Start timer (for calculating frame rate)
            t1 = cv2.getTickCount()

            # Grab frame from video stream
            frame1 = videostream.read()
            isVideo = True

            # Acquire frame and resize to expected shape [1xHxWx3]
            frame = frame1.copy()
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, (width, height))
            input_data = np.expand_dims(frame_resized, axis=0)

            # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
            if floating_model:
                input_data = (np.float32(input_data) - input_mean) / input_std

            # Perform the actual detection by running the model with the image as input
            interpreter.set_tensor(input_details[0]['index'],input_data)
            interpreter.invoke()

            # Retrieve detection results
            boxes = interpreter.get_tensor(output_details[0]['index'])[0] # Bounding box coordinates of detected objects
            classes = interpreter.get_tensor(output_details[1]['index'])[0] # Class index of detected objects
            scores = interpreter.get_tensor(output_details[2]['index'])[0] # Confidence of detected objects
            #num = interpreter.get_tensor(output_details[3]['index'])[0]  # Total number of detected objects (inaccurate and not needed)
            isPerson = False
            # Loop over all detections and draw detection box if confidence is above minimum threshold
            for i in range(len(scores)):
                if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):

                    # Get bounding box coordinates and draw box
                    # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
                    ymin = int(max(1,(boxes[i][0] * imH)))
                    xmin = int(max(1,(boxes[i][1] * imW)))
                    ymax = int(min(imH,(boxes[i][2] * imH)))
                    xmax = int(min(imW,(boxes[i][3] * imW)))
                    
                    cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)

                    # Draw label
                    object_name = labels[int(classes[i])] # Look up object name from "labels" array using class index
                    label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
                    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
                    label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
                    cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
                    cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text

                    if object_name == "person" and int(scores[i]*100 > 50):
                        isPerson = True
                        break
                    else: 
                        print(object_name)
                        isPerson = False

                    
            # Draw framerate in corner of frame
            cv2.putText(frame,'FPS: {0:.2f}'.format(frame_rate_calc),(30,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)
            # Calculate framerate
            t2 = cv2.getTickCount()
            time1 = (t2-t1)/freq
            frame_rate_calc= 1/time1

            if isPerson:
                # Person found -> apply for the first time, then do not apply till they disapper. if they disappear
                if personFound:
                    if isTimer:
                        print(f"person found again before timer passed reset {RESET_SEC} sec")
                        timer.reset(RESET_SEC)
                    else:
                        print("person found again",isPerson,personFound,isTimer)
                else:
                    print("First person found! call applyData, isTimer:",isTimer)
                    applyLightData(eval(current_lightData))
                    personFound = True

                # if person not found wait for 5 sec, and apply.
                # if new person appears then apply 5 sec again.
            else:
                if not personFound:
                    print("Just Waiting...")

                else :
                    if not isTimer:
                        isTimer = True
                        timer = TimerReset(RESET_SEC, applyLightData, args=[eval(current_lightData)], kwargs={})
                        timer.start()
                        print(f"person not found! wait {RESET_SEC} sec")
                    else:
                        print("waiting for timer to finish...")
                # if person not found wait for 5 sec, and apply.
                # if new person appears then apply 5 sec again.
            cv2.imwrite("frame.jpg",frame)

        videostream.stop()

isAuto =True
isLightSensor = True
lightValue = 80
lightSensorValue = 0

data = {"isAuto":isAuto,"isLightSensor":isLightSensor,"lightValue":lightValue,"lightSensorValue":lightSensorValue}

l = LightSensor()
l.start()

class CamHandler(BaseHTTPRequestHandler):
    def _send_cors_headers(self):
        """ Sets headers required for CORS """
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "x-api-key,Content-Type")

    def send_dict_response(self, d):
        """ Sends a dictionary (JSON) back to the client """
        self.wfile.write(bytes(dumps(d), "utf8"))

    def do_OPTIONS(self):
        self.send_response(200)
        self._send_cors_headers()
        self.end_headers()

    print("Cam handler")
    # Initialize video stream
    def do_GET(self):
        global data
        global isAuto
        global isLightSensor
        global lightValue
        global lightSensorValue

        parsed_path = urlparse(self.path)
        print(data)
        send_data = json.dumps(data)
        if(parsed_path.path =='/data'):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(bytes(send_data, 'utf-8') )


        if self.path.endswith('.jpg'):
            self.path = './frame.jpg'
            try:
                f = open(self.path,'rb') #self.path has /test.html
#note that this potentially makes every file on your computer readable by the internet

                self.send_response(200)
                self.send_header('Content-type','image/jpeg')
                self.end_headers()
                self.wfile.write(f.read())
                f.close()
                return

            except IOError:
                self.send_error(404,'File Not Found: %s' % self.path)

        if self.path.endswith('.mjpg'):
            self.send_response(200)
            self.send_header(
                'Content-type',
                'multipart/x-mixed-replace; boundary=--jpgboundary'
            )
            self.end_headers()
            while True:
                global imW, imH
                global frame
                global isVideo
                global frame_rate_calc
                if isVideo:
                    try:
                        img_str = cv2.imencode('.jpg', frame)[1].tostring()
                        self.send_header('Content-type', 'image/jpeg')
                        self.end_headers()
                        self.wfile.write(img_str)
                        self.wfile.write(b"\r\n--jpgboundary\r\n")
                    except KeyboardInterrupt:
                        self.wfile.write(b"\r\n--jpgboundary--\r\n")
                        break
                    except BrokenPipeError:
                        continue
                else:
                    try:
                        # gif 처리
                        gif = cv2.VideoCapture('waiting.mp4')
                        ret, gif_frame = gif.read()  # ret=True if it finds a frame else False.
                        frame_counter = 0
                        print(imW,imH)
                        gif.set(3,imW)
                        gif.set(4,imH)

                        while ret:
                            ret, gif_frame = gif.read()
                            gif_frame = gif_frame.copy()
                            gif_frame = cv2.resize(gif_frame, (imW,imH))
                            # something to do 'frame'
                            # ...
                            # 다음 frame 읽음
                            if frame_counter == gif.get(cv2.CAP_PROP_FRAME_COUNT):
                                frame_counter = 0 #Or whatever as long as it is the same as next line

                            gif.set(cv2.CAP_PROP_POS_FRAMES, frame_counter)
                            print("Loading...")
                            # Grab frame from video stream
                            img_str = cv2.imencode('.jpg', gif_frame)[1].tostring()
                            self.send_header('Content-type', 'image/jpeg')
                            self.end_headers()
                            self.wfile.write(img_str)
                            self.wfile.write(b"\r\n--jpgboundary\r\n")
                            frame_counter += 1
                            #setLight(frame_counter*10)
                            if isVideo:
                                break

                    except Exception as e:
                        print(e)
                        return None
            return

        if self.path.endswith('.html'):
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(b'<html><head></head><body>')
            self.wfile.write(b'<img src="http://192.168.124.41:1443/cam.mjpg"/>')
            self.wfile.write(b'</body></html>')
            return

    def do_POST(self):
        try:
            global data
            global isAuto
            global isLightSensor
            global lightValue
            global lightSensorValue

            if self.path.endswith("/mode"):
                ctype, pdict = cgi.parse_header(self.headers['content-type'])
                pdict['boundary'] = bytes(pdict['boundary'], "utf-8")
                if ctype == 'multipart/form-data':
                    fields = cgi.parse_multipart(self.rfile, pdict)
                    print("Fields value is", fields)
                    value = fields.get('type')
                    value = value[0].decode("utf-8")

                    if value == "manual":
                        isAuto = False
                        data["isAuto"] = isAuto
                        print("Change to manual mode")

                    elif value == "auto":
                        isAuto = True
                        data["isAuto"] = isAuto
                        print("Change to auto mode")
                    
                    elif value == "light_sensor":
                        isLightSensor = True
                        data["isLightSensor"] = isLightSensor
                        print("apply Light sensor")

                    elif value == "no_light_sensor":
                        isLightSensor = False
                        data["isLightSensor"] = isLightSensor
                        print("not apply Light sensor")
                    else:
                        print("unknown vlaue")

                    print("value applied is ", value)
            
                    print(data)

                    self.send_response(200)
                    self.send_header('Content-Type', 'application/text')
                    self.end_headers()
                    self.wfile.write(value.encode(encoding='utf_8'))

            if self.path.endswith("/light"):
                ctype, pdict = cgi.parse_header(self.headers['content-type'])
                pdict['boundary'] = bytes(pdict['boundary'], "utf-8")
                if ctype == 'multipart/form-data':
                    fields = cgi.parse_multipart(self.rfile, pdict)
                    print("Fields value is", fields)
                    value = fields.get('value')
                    value = value[0].decode("utf-8")
                    print("value applied is ", value)
                    lightValue = value
                    data["lightValue"] = lightValue
                    setLight(lightValue)
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/text')
                    self.end_headers()
                    self.wfile.write(value.encode(encoding='utf_8'))

        except:
            self.send_response(400)
            self.send_header('Content-Type', 'application/text')
            self.end_headers()
            returnVal = 'error'
            self.wfile.write(returnVal.encode(encoding='utf_8'))

class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """Handle requests in a separate thread."""

if __name__ == '__main__':
    server = ThreadedHTTPServer(('192.168.124.41', 1443), CamHandler)
    # server.socket = ssl.wrap_socket(server.socket,
    # keyfile="keys/private.key", 
    # certfile='keys/certificate.cert', server_side=True)
    initDetector().start()

    try:
        print("server started at https://192.168.124.41:1443/cam.html")
        server.serve_forever()

    except KeyboardInterrupt:
        pass
        #pwm_led.stop()
        #GPIO.cleanup()
        sys.exit()
        os._exit(0)
        server.socket.close()
    myLogger.info("Started!")