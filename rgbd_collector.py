#!/usr/bin/env python3

import cv2
import depthai as dai
import numpy as np
from itertools import cycle
from pathlib import Path
import time
import queue

def clamp(num, v0, v1):
    return max(v0, min(num, v1))

# Closer-in minimum depth, disparity range is doubled (from 95 to 190):
extended_disparity = True
# Better accuracy for longer distance, fractional disparity 32-levels:
subpixel = False
# Better handling for occlusions:
lr_check = True

# Create pipeline
pipeline = dai.Pipeline()

# Define sources and outputs
camRgb = pipeline.create(dai.node.ColorCamera)
monoLeft = pipeline.create(dai.node.MonoCamera)
monoRight = pipeline.create(dai.node.MonoCamera)
depth = pipeline.create(dai.node.StereoDepth)

#create XLink output
xout = pipeline.create(dai.node.XLinkOut)
xout.setStreamName("frames")

#properties

camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
camRgb.setPreviewSize(400, 400)

camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_800_P)
#camRgb.setPreviewSize(400, 400)
camRgb.setBoardSocket(dai.CameraBoardSocket.RGB)
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
camRgb.setPreviewSize(400, 400)
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_800_P)
#camRgb.setPreviewSize(400, 400)
camRgb.setBoardSocket(dai.CameraBoardSocket.RGB)

# Properties
monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_800_P)
monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_800_P)
monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_800_P)
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_800_P)
monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)

# Create a node that will produce the depth map (using disparity output as it's easier to visualize depth this way)
depth.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
# Options: MEDIAN_OFF, KERNEL_3x3, KERNEL_5x5, KERNEL_7x7 (default)
depth.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)
depth.setLeftRightCheck(lr_check)
depth.setExtendedDisparity(extended_disparity)
depth.setSubpixel(subpixel)

# Linking
monoLeft.out.link(depth.left)
monoRight.out.link(depth.right)

camRgb.preview.link(xout.input)
depth.disparity.link(xout.input)

q = queue.Queue()

def newFrame(inFrame):
    global q
    num = inFrame.getInstanceNum()
    name = "color" if num ==0 else "depth"

    frame = inFrame.getCvFrame()
    q.put({"name":name, "frame":frame})

# Connect to device and start pipeline
with dai.Device(pipeline, usb2Mode=True) as device:
    # Output queue will be used to get the disparity frames from the outputs defined above

    device.getOutputQueue(name="frames", maxSize=4, blocking=False).addCallback(newFrame)

    dirName = "testdata"

    Path(dirName).mkdir(parents=True,exist_ok=True)
    i = 1
    j = 0
    while True:
        data = q.get()  # blocking call, will wait until a new data has arrived
        cv2.imshow(data["name"],data["frame"])
        #time.sleep(.5)
        if data['name'] == "color":
            if j==0:
                pass
            else:
                cv2.imwrite(f"{dirName}/{int(i/100)}{data['name']}.png", data["frame"])
            j = 0
        else:
            if j==1:
                pass
            else:
                cv2.imwrite(f"{dirName}/{int(i/100)}{data['name']}.png", data["frame"])
            j=1
        i+=1

        if cv2.waitKey(1) == ord('q'):
            break