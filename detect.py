# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either expess or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Main script to run the object detection routine."""
import argparse
import sys
import cv2
from tflite_support.task import core
from tflite_support.task import processor
from tflite_support.task import vision
import utils
import os
from os.path import exists
import asyncio
import time
import datetime
import requests
import json
import RPi.GPIO as GPIO

millis = lambda: int(round(time.time() * 1000))
TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ0eXBlIjoiYWRtaW4iLCJfaWQiOiI2NDE1YzQzYTM2YzY2ZWFhODVjMzg0ZjEiLCJpYXQiOjE2ODQ4Mzc1NzgsImV4cCI6MTY4NTA5Njc3OH0.C5Xslhx33kxY2PvYKhX-rbNbaYLJMF9XQe9YCrx491c"
LED_OUT = 23

GPIO.setmode(GPIO.BCM)
GPIO.setup(LED_OUT, GPIO.OUT)
GPIO.output(LED_OUT, GPIO.LOW)

async def portal_connect():
  print('Connecting to portal...')
  res = requests.get('https://security-portal.onrender.com/')
  if res.status_code == 200:
    print('Connected to portal')
  else:
    print('Failed to connect')
    

async def update_portal(plate):
  time = datetime.datetime.now()
  date = time.replace(minute=0, hour=0, second=0, microsecond=0000)
  registered = 0
  
  body = {
    "plateNumber": plate,
    "date": str(date),
    "time": str(time)
  }
    
  header = { 
    "Authorization" : f'Bearer {TOKEN}',
    "Content-Type": 'application/json',
  }
  
  try:
    response = requests.post('https://security-portal.onrender.com/activities/upsert', 
    json=body, 
    headers=header)
    
    response = response.json()
    
    if "vehicle_id" in response['message']:
      return 1
    elif "acknowledged" in response['message']:
      return 3
    
    print(response['message'])
    return 2;
      
  except Exception as e:
    print(e) 
    return 0
  
  
async def get_plate():
    try:
      regions = ['us-ca'] # Change to your country
      with open('output/output.jpg', 'rb') as fp:
        response = requests.post(
          'https://api.platerecognizer.com/v1/plate-reader/',
          data=dict(regions=regions),  # Optional
          files=dict(upload=fp),
          headers={'Authorization': 'Token 7611813154add04c050a3388de52068bfff1f96a'})

      if response.status_code == 201:
        
        result = response.json()['results']
        
        if len(result) > 0: 
          plate = response.json()['results'][0]['plate'].upper()
          print(plate)
          return await update_portal(plate)
        else:
          return 0
      
      print(response.json())
      
    except Exception as e:
      print(e)

async def run(model: str, camera_id: int, width: int, height: int, num_threads: int,
        enable_edgetpu: bool) -> None:
  """Continuously run inference on images acquired from the camera.

  Args:
    model: Name of the TFLite object detection model.
    camera_id: The camera id to be passed to OpenCV.
    width: The width of the frame captured from the camera.
    height: The height of the frame captured from the camera.
    num_threads: The number of CPU threads to run the model.
    enable_edgetpu: True/False whether the model is a EdgeTPU model.
  """
    
  output_path = 'output/'
  ct1 = 0
  ct2 = 0
  registered = 0
  
  # Variables to calculate FPS
  counter, fps = 0, 0
  start_time = time.time()

  # Start capturing video input from the camera
  cap = cv2.VideoCapture(camera_id)
  cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
  cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

  # Visualization parameters
  row_size = 20  # pixels
  left_margin = 24  # pixels
  text_color = (0, 255, 0)  # green
  font_size = 1
  font_thickness = 1
  fps_avg_frame_count = 10

  # Initialize the object detection model
  base_options = core.BaseOptions(
      file_name=model, use_coral=enable_edgetpu, num_threads=num_threads)
  detection_options = processor.DetectionOptions(
      max_results=1, score_threshold=0.45)
  options = vision.ObjectDetectorOptions(
      base_options=base_options, detection_options=detection_options)
  detector = vision.ObjectDetector.create_from_options(options)

  # Continuously capture images from the camera and run inference
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      sys.exit(
          'ERROR: Unable to read from webcam. Please verify your webcam settings.'
      )

    counter += 1
    image = cv2.flip(image, 1)

    # Convert the image from BGR to RGB as required by the TFLite model.
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Create a TensorImage object from the RGB image.
    input_tensor = vision.TensorImage.create_from_array(rgb_image)

    # Run object detection estimation using the model.
    detection_result = detector.detect(input_tensor)

    
    if len(detection_result.detections) > 0:
        
        conf = detection_result.detections[0].categories[0].score       
        if conf >= 0.50 :
            image1 = cv2.flip(image, 1)
      
            file_exists = exists(f'{output_path}output.jpg')
            if millis() > (ct1 + 5000):
              
              print('Detection')
            
              if file_exists:
                  os.remove(f'{output_path}output.jpg')

              cv2.imwrite(f'{output_path}output.jpg', image1)
              registered = await get_plate()
              
              if registered == 1:
                print('Registered')
              elif registered == 2:
                print('Not registered')
              elif registered == 3:
                print('Vehicle out')
              else:
                print('No plate detected.')
                
              ct1 = millis()
              ct2 = millis()
    
    if registered == 1:
      print('LED ON')
      GPIO.output(LED_OUT, GPIO.HIGH)
      if millis() > (ct2 + 6000):
        registered = 0
        GPIO.output(LED_OUT, GPIO.LOW)
        print('LED OFF')
      

    # Draw keypoints and edges on input image
    image = utils.visualize(image, detection_result)


    # Calculate the FPS
    if counter % fps_avg_frame_count == 0:
      end_time = time.time()
      fps = fps_avg_frame_count / (end_time - start_time)
      start_time = time.time()

    # Show the FPS
    fps_text = 'FPS = {:.1f}'.format(fps)
    text_location = (left_margin, row_size)
    cv2.putText(image, fps_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                font_size, text_color, font_thickness)

    # Stop the program if the ESC key is pressed.
    if cv2.waitKey(1) == 27:
      break
    cv2.imshow('License Plate Detection', image)

  cap.release()
  GPIO.cleanup()
  cv2.destroyAllWindows()


async def main():
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument(
      '--model',
      help='Path of the object detection model.',
      required=False,
      default='cm_detect.tflite')
  parser.add_argument(
      '--cameraId', help='Id of camera.', required=False, type=int, default=0)
  parser.add_argument(
      '--frameWidth',
      help='Width of frame to capture from camera.',
      required=False,
      type=int,
      default=640)
  parser.add_argument(
      '--frameHeight',
      help='Height of frame to capture from camera.',
      required=False,
      type=int,
      default=480)
  parser.add_argument(
      '--numThreads',
      help='Number of CPU threads to run the model.',
      required=False,
      type=int,
      default=4)
  parser.add_argument(
      '--enableEdgeTPU',
      help='Whether to run the model on EdgeTPU.',
      action='store_true',
      required=False,
      default=False)
  args = parser.parse_args()
  
  await portal_connect()
  await run(args.model, int(args.cameraId), args.frameWidth, args.frameHeight,
      int(args.numThreads), bool(args.enableEdgeTPU))
 
"""
if __name__ == '__main__':
  main()
"""

asyncio.run(main())
