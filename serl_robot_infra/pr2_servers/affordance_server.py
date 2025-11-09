"""
Affordance server for the robot. This server is responsible for providing the robot with affordances
"""

import numpy as np
from scipy.spatial.transform import Rotation as R

import rospy
from geometry_msgs.msg import PoseArray

class AffordanceServer:
    def __init__(self):
        super().__init__()
        self.sub_pre_traj = rospy.Subscriber(
            "/affordance/pre_traj", PoseArray, self._pre_traj_callback
        )
        self.sub_post_traj = rospy.Subscriber(
            "/affordance/post_traj", PoseArray, self._post_traj_callback
        )

        # placeholder
        # pose : xyz + rpy
        self.pre_traj = None
        self.post_traj = None
        self.grasp_pose = None

    def _pre_traj_callback(self, msg):
        pre_traj = []
        for pose in msg.poses:
            xyz = np.array([pose.position.x, pose.position.y, pose.position.z])
            xyzw = np.array(
                [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
            )
            pose = np.concatenate([xyz, R.from_quat(xyzw).as_euler('xyz')])
            pre_traj.append(pose)
        self.pre_traj = np.array(pre_traj[::-1]) # (N, 4, 4)
        self.grasp_pose = self.pre_traj[-1] # (4, 4)

    def _post_traj_callback(self, msg):
        post_traj = []
        for pose in msg.poses:
            xyz = np.array([pose.position.x, pose.position.y, pose.position.z])
            xyzw = np.array(
                [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
            )
            pose = np.concatenate([xyz, R.from_quat(xyzw).as_euler('xyz')])
            post_traj.append(pose)
        self.post_traj = np.array(post_traj) # (N, 4, 4)
        # self.grasp_pose = self.post_traj[0] # (4, 4)
