import numpy as np

import rospy
from sensor_msgs.msg import Joy

MAX_TRANSLATION = 0.68359375
MAX_ROTATION = 0.68359375

class SpacenavServer:
    def __init__(self):
        super().__init__()
        self.translation = np.zeros(3)
        self.rotation = np.zeros(3)
        self.buttons = [0.0, 0.0]

        self.sub_spacenav_state = rospy.Subscriber(
            "/spacenav/joy",
            Joy,
            self._update_spacenav_state
        )

    def _update_spacenav_state(self, msg):
        self.translation = np.array(msg.axes[:3]) / MAX_TRANSLATION # translation xyz
        self.rotation = np.array(msg.axes[3:]) / MAX_ROTATION       # rotation xyz
        self.buttons = list(msg.buttons)                            # buttons
