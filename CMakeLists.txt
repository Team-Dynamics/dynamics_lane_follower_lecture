cmake_minimum_required(VERSION 2.8.3)
project(lane_follower_lecture)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  sensor_msgs
  std_msgs
)

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES lane_detection
#  CATKIN_DEPENDS cv_bridge opencv2 roscpp rospy sensor_msgs std_msgs
#  DEPENDS system_lib
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)
