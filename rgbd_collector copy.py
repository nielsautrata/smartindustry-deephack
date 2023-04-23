#!/usr/bin/env python3

import cv2
import depthai as dai
import numpy as np
from itertools import cycle
from pathlib import Path
import time
import queue


# Weights to use when blending depth/rgb image (should equal 1.0)
rgbWeight = 0.4
depthWeight = 0.6

msgs = dict()

def add_msg(msg, name, seq = None):
    if seq is None:
        seq = msg.getSequenceNum()
    seq = str(seq)
    if seq not in msgs:
        msgs[seq] = dict()
    msgs[seq][name] = msg

def get_msgs():
    global msgs
    seq_remove = [] # Arr of sequence numbers to get deleted
    for seq, syncMsgs in msgs.items():
        seq_remove.append(seq) # Will get removed from dict if we find synced msgs pair
        # Check if we have both detections and color frame with this sequence number
        if len(syncMsgs) == 2: # rgb + depth
            for rm in seq_remove:
                del msgs[rm]
            return syncMsgs # Returned synced msgs
    return None


def updateBlendWeights(percent_rgb):
    """
    Update the rgb and depth weights used to blend depth/rgb image

    @param[in] percent_rgb The rgb weight expressed as a percentage (0..100)
    """
    global depthWeight
    global rgbWeight
    rgbWeight = float(percent_rgb)/100.0
    depthWeight = 1.0 - rgbWeight






print("testing")
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
#depthout = pipeline.create(dai.node.XLinkOut)
#depthout.setStreamName("depthframes")

#properties
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
camRgb.setPreviewSize(400, 400)
camRgb.setFps(10)
# Properties
monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)
monoLeft.setFps(10)
monoRight.setFps(10)
# Create a node that will produce the depth map (using disparity output as it's easier to visualize depth this way)
depth.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
# Options: MEDIAN_OFF, KERNEL_3x3, KERNEL_5x5, KERNEL_7x7 (default)
depth.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)
depth.setLeftRightCheck(lr_check)
depth.setDepthAlign(dai.CameraBoardSocket.RGB)
depth.setExtendedDisparity(extended_disparity)
depth.setSubpixel(subpixel)

# Linking
monoLeft.out.link(depth.left)
monoRight.out.link(depth.right)

camRgb.preview.link(xout.input)
depth.disparity.link(xout.input)

q = queue.Queue()
matching_dictionary = {}
def newFrame(inFrame):
    global q
    global matching_dictionary

    num = inFrame.getInstanceNum()
    tim = inFrame.getTimestamp()
    num_seq = inFrame.getSequenceNum()
    if num_seq%2 !=0:
        num_seq-=1
    num_seq = str(num_seq)
    name = "color" if num ==0 else "depth" 
    print(num)
    frame = inFrame.getCvFrame()
    if num_seq not in matching_dictionary:
        matching_dictionary[num_seq] = []
    matching_dictionary[num_seq].append({"name":name,"time":tim})# "frame":frame
    if len(matching_dictionary[num_seq]) >1:
        q.put(matching_dictionary[num_seq])

    #q.put({"name":name, "frame":frame})

# Connect to device and start pipeline
with dai.Device(pipeline, usb2Mode=True) as device:
    # Output queue will be used to get the disparity frames from the outputs defined above
    device.getOutputQueue(name="frames", maxSize=4, blocking=False).addCallback(newFrame)    

    dirName = "maybe_this_will_work"
    Path(dirName).mkdir(parents=True,exist_ok=True)
    i = 1
    j = 0
    while True:
        data = q.get()  # blocking call, will wait until a new data has arrived
        #cv2.imshow(data["name"],data["frame"])
        #time.sleep(.5)
        print(data)
        #for key,val in matching_dictionary.items():
        #    if len(val) >2:
        #        print(key, val)
                #del matching_dictionary[key]

        ''' if data['name'] == "color":
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
        '''
        if cv2.waitKey(1) == ord('q'):
            break