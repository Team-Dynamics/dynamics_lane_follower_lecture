#!/usr/bin/env python2
import rospy
import cv2
from helpers import imgmsg_to_cv2, cv2_to_imgmsg, parse_hsv_string, PID
import message_filters
from sensor_msgs.msg import Image, LaserScan
from geometry_msgs.msg import Twist

class Follower:
    def __init__(self, color_lower_thresh, color_upper_thresh, topic_cmd_vel, 
                    stopping_threshold, cmd_vel_speed, kP, kI, pub_visual_output):
        
        self.color_lower_thresh = color_lower_thresh
        self.color_upper_thresh = color_upper_thresh
        self.stopping_threshold = stopping_threshold
        self.cmd_vel_speed = cmd_vel_speed
        self.PID = PID(kP, kI, 0)
        
        self.pub_visual_output = pub_visual_output
        self.pub_cmd_vel = rospy.Publisher(topic_cmd_vel, Twist, queue_size=1)
        
        if pub_visual_output:
            self.pub_image = rospy.Publisher('/output_image_lane_follower', Image, queue_size=1)
            self.pub_segmenation_map = rospy.Publisher('/output_binary_map_lane_follower', Image, queue_size=1)
        
    def callback(self, data_sub_cam, data_sub_scan):
        msg_twist = Twist()

        # Look at 15 data points to reduce the probability of getting an invalid one
        for i in range(15):
            obstacle_distance = data_sub_scan.ranges[i]        
            if(obstacle_distance != 0):
                break
        rospy.loginfo("Obstacle distance: " + str(obstacle_distance))

        image = imgmsg_to_cv2(data_sub_cam)
        image = cv2.resize(image, (640, 480), interpolation=cv2.INTER_LINEAR)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Calculate the binary segmentation mask through OpenCV inRange
        mask = cv2.inRange(hsv, self.color_lower_thresh, self.color_upper_thresh)
        
        # Select the region of the obtained binary mask with which to proceed
        h, w, _ = image.shape
        search_top = h - int(h/2)
        search_bot = h
        mask[0:search_top, 0:w] = 0
        mask[search_bot:h, 0:w] = 0

        # Calculate the centroid pixel coordinate of the selected region of the binary mask through OpenCV moments
        M = cv2.moments(mask)

        if M['m00'] > 0:
            cx = int(M['m10']/M['m00']) # x coordinate of the centroid pixel
            cy = int(M['m01']/M['m00']) # y coordinate of the centroid pixel
            cv2.circle(image, (cx, cy), 20, (0, 0, 255), -1) # Draw a circle on the original image at the centroid position

            # The error is the difference between the horizontal image center coordinate and the centroid's x coordinate
            err = (cx - w/2.0)/(w/2.0) * 100.0 # Normalize the error to a range between 0 and 100
            msg_twist.angular.z = -self.PID.update(err) # Putting the error into a PI controller
        else:
            self.PID.reset() # Reset the PID controller so that the error does not grow when the robot is not on the line

        msg_twist.linear.x = self.cmd_vel_speed

        # Set the velocities to 0 if an obstacle is closer than the stopping_threshold
        # The range is 0 if a measurement is invalid or the distance is bigger than the sensor's max range
        if (obstacle_distance < self.stopping_threshold and obstacle_distance > 0):
            msg_twist.linear.x = 0
            msg_twist.angular.z = 0
            self.PID.reset() # Reset the PID controller so that the error does not grow while the robot is standing

        if self.pub_visual_output:
            seg_mask_pub = cv2_to_imgmsg(mask, encoding='mono8')
            self.pub_segmenation_map.publish(seg_mask_pub)

            image_pub = cv2_to_imgmsg(image, encoding='bgr8')
            self.pub_image.publish(image_pub)

        self.pub_cmd_vel.publish(msg_twist)

    def shut_down_autonomy(self):
        # Publish a Twist message with all velocities set to 0 to stop the movement of the robot when the node is shut down
        msg_twist = Twist()
        self.pub_cmd_vel.publish(msg_twist)
        rospy.loginfo("Sent stopping message.")

if __name__ == '__main__':
    rospy.init_node('lane_follower_lecture', anonymous=True)

    TOPIC_INPUT_IMAGE = rospy.get_param('~topic_input_image')
    TOPIC_INPUT_SCAN = rospy.get_param('~topic_input_scan')
    TOPIC_OUTPUT_CMD_VEL = rospy.get_param('~topic_output_cmd_vel')
    COLOR_LOWER_THRESH = rospy.get_param('~segmentation_color_lower_thresh_hsv')
    COLOR_UPPER_THRESH = rospy.get_param('~segmentation_color_upper_thresh_hsv')
    STOPPING_THRESH = rospy.get_param('~stopping_thresh')
    CMD_VEL_SPEED = rospy.get_param('~cmd_vel_speed')
    CONTROLLER_KP = rospy.get_param('~controller_kP')
    CONTROLLER_KI = rospy.get_param('~controller_kI')
    PUB_VISUAL_OUTPUT = rospy.get_param('~pub_visual_output')
    
    follower = Follower(parse_hsv_string(COLOR_LOWER_THRESH), parse_hsv_string(COLOR_UPPER_THRESH),
                        TOPIC_OUTPUT_CMD_VEL, STOPPING_THRESH, CMD_VEL_SPEED, CONTROLLER_KP, 
                        CONTROLLER_KI, PUB_VISUAL_OUTPUT)

    # Use a message filter to process messages from different topics in one callback function
    sub_cam = message_filters.Subscriber(TOPIC_INPUT_IMAGE, Image)
    sub_scan = message_filters.Subscriber(TOPIC_INPUT_SCAN, LaserScan)
    ts = message_filters.ApproximateTimeSynchronizer([sub_cam, sub_scan], 1, 0.4)
    ts.registerCallback(follower.callback)
    
    rospy.on_shutdown(follower.shut_down_autonomy)

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")