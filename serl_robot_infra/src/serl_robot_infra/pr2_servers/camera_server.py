"""
Camera server for PR2
"""

import numpy as np

import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CompressedImage, CameraInfo


class CameraServer:
    def __init__(self):
        super().__init__()
        self.bridge = CvBridge()
        self.sub_image = rospy.Subscriber(
            "/kinect_head/rgb/image_rect_color/compressed",
            CompressedImage,
            self._update_image
        )
        self.sub_depth = rospy.Subscriber(
            "/kinect_head/depth_registered/image_rect",
            Image,
            self._update_depth
        )
        self.sub_camera_info = rospy.Subscriber(
            "/kinect_head/rgb/camera_info",
            CameraInfo,
            self._update_camera_info
        )

    def _update_image(self, msg):
        self.image = self.bridge.compressed_imgmsg_to_cv2(msg) # bgr8

    def _update_depth(self, msg):
        self.depth = np.array(self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough"), dtype=np.float32).reshape(480, 640)

    def _update_camera_info(self, msg):
        self.width = msg.width
        self.height = msg.height
        self.K = np.array(msg.K).reshape(3, 3)
