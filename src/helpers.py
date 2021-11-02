#!/usr/bin/env python2
from sensor_msgs.msg import Image
import numpy as np
import rospy
import time

def parse_hsv_string(hsv_string):
    """
    Parses strings to hsv lists adhering to opencv hsv standards.
    """
    hsv_sub_strings = hsv_string.split("-")    
    
    if(len(hsv_sub_strings) != 3):
        raise ValueError("HSV values invalid!")
    
    hsv = []
    for sub_str in hsv_sub_strings:
        hsv.append(int(sub_str))

    if(hsv[0] > 360 or hsv[0] < 0):
        raise ValueError("HSV values invalid!")

    if(hsv[1] > 100 or hsv[1] < 0):
        raise ValueError("HSV values invalid!")

    if(hsv[2] > 100 or hsv[2] < 0):
        raise ValueError("HSV values invalid!")
  
    return np.array([int(hsv[0]/2), int((255/100) * hsv[1]), int((255/100) * hsv[2])])

class PID:
    """
        PID controller taken from: https://github.com/korfuri/PIDController
    """
    def __init__(self, Kp, Ki, Kd, origin_time=None):
        if origin_time is None:
            origin_time = time.time()

        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd

        self.Cp = 0.0
        self.Ci = 0.0
        self.Cd = 0.0

        self.previous_time = origin_time
        self.previous_error = 0.0

    def update(self, error, current_time=None):
        if current_time is None:
            current_time = time.time()
        dt = current_time - self.previous_time
        if dt <= 0.0:
            return 0
        de = error - self.previous_error

        self.Cp = error
        self.Ci += error * dt
        self.Cd = de / dt
        
        self.previous_time = current_time
        self.previous_error = error

        return (
            (self.Kp * self.Cp)  
            + (self.Ki * self.Ci) 
            + (self.Kd * self.Cd)
        )

    def reset(self):
        self.previous_error = 0.0

def imgmsg_to_cv2(data, desired_encoding="passthrough", flip_channels=False):
    """
    Converts a ROS image to an OpenCV image without using the cv_bridge package.
    """

    if desired_encoding == "passthrough":
        encoding = data.encoding
    else:
        encoding = desired_encoding

    if encoding == 'bgr8' or (encoding=='rgb8' and flip_channels):
        return np.frombuffer(data.data, np.uint8).reshape((data.height, data.width, 3))
    elif encoding == 'bgra8': 
        return np.frombuffer(data.data, np.uint8).reshape((data.height, data.width, 4))[...,:3]
    elif encoding == 'rgb8' or (encoding=='bgr8' and flip_channels):
        return np.frombuffer(data.data, np.uint8).reshape((data.height, data.width, 3))[:, :, ::-1]
    elif encoding == 'mono8' or encoding == '8UC1':
        return np.frombuffer(data.data, np.uint8).reshape((data.height, data.width))
    elif encoding == 'mono16' or encoding == '16UC1':
        return np.frombuffer(data.data, np.uint16).reshape((data.height, data.width))
    else:
        rospy.logwarn("Unsupported encoding %s" % encoding)
        return None

def cv2_to_imgmsg(cv2img, encoding='bgr8'):
    """
    Converts an OpenCV image to a ROS image without using the cv_bridge package.
    """

    BPP = {
        'bgr8': 3,
        'rgb8': 3,
        '16UC1': 2,
        '8UC1': 1,
        'mono16': 2,
        'mono8': 1
    }

    msg = Image()
    msg.width = cv2img.shape[1]
    msg.height = cv2img.shape[0]
    msg.encoding = encoding
    msg.step = BPP[encoding]*cv2img.shape[1]
    msg.data = np.ascontiguousarray(cv2img).tobytes()

    return msg