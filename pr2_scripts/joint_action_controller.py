#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
import numpy as np
import skrobot
from scipy.spatial.transform import Rotation as R

import rospy
from std_msgs.msg import Header
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from serl_controllers_msgs.msg import ControllerState


class PR2JointActionController(object):
    """
    PR2JointActionController using skrobot API

    """

    def __init__(self, arm: str = "rarm", rate: float = 30.0):
        self.robot = skrobot.models.PR2()
        self.interface = skrobot.interfaces.PR2ROSRobotInterface(self.robot)
        self.arm = self.robot.rarm if arm == "rarm" else self.robot.larm
        self.controller_type = f"{arm}_controller"
        time.sleep(1)

        # Subscriber for pose command
        self.sub_pose_command = rospy.Subscriber(
            "/joint_action_controller/pose_command",
            PoseStamped,
            self.pose_command_callback,
            queue_size=1,
        )
        self.sub_joint_command = rospy.Subscriber(
            "/joint_action_controller/joint_command",
            JointState,
            self.joint_command_callback,
            queue_size=1,
        )

        # Publisher for robot state
        self.pub_controller_state = rospy.Publisher(
            "/joint_action_controller/joint_controller_state",
            ControllerState,
            queue_size=1,
        )

        # Timer
        timer = rospy.Timer(rospy.Duration(1.0 / rate), self.timer_callback)

    def pose_command_callback(self, msg: PoseStamped) -> None:
        """
        Get Command from PoseStamped message and send command to robot

        Args:
            msg (PoseStamped): PoseStamped message, which contains End-effector pose command
                pose (Pose): Pose message
                    position (Point): Position to command
                        x (float): x position
                        y (float): y position
                        z (float): z position
                    orientation (Quaternion): Orientation to command
                        w (float): w orientation
                        x (float): x orientation
                        y (float): y orientation
                        z (float): z orientation
        """
        # self.interface.update_robot_state()
        # self.robot.angle_vector(self.interface.potentio_vector())
        pose = msg.pose
        xyz = np.array([pose.position.x, pose.position.y, pose.position.z])
        wxyz = np.array(
            [
                pose.orientation.w,
                pose.orientation.x,
                pose.orientation.y,
                pose.orientation.z,
            ]
        )
        wxyz = wxyz / np.linalg.norm(wxyz)
        target_coords = skrobot.coordinates.Coordinates(
            pos=xyz,
            rot=wxyz,
        )
        start_time = time.time()
        self.robot.inverse_kinematics(
            target_coords,
            link_list=self.arm.link_list,
            move_target=self.arm.end_coords,
            rotation_axis=True,
            revert_if_fail=False,
            stop=5,
        )
        print(f"IK computation time: {time.time() - start_time:.4f} sec")

        # Only update the arm joints, keep other joints unchanged
        # current_av = self.interface.potentio_vector()  # Get current robot state
        # desired_av = current_av.copy()  # Start with current state

        # # Update only the arm joint angles from IK result
        # arm_av = self.arm.angle_vector()  # Get IK result for arm
        # for i, joint_name in enumerate(self.arm.joint_names):
        #     joint_id = self.robot.joint_names.index(joint_name)
        #     desired_av[joint_id] = arm_av[i]

        # controller_type = [self.interface.rarm_controller] if self.arm == self.robot.rarm else [self.interface.larm_controller]
        # self.interface.angle_vector(desired_av, time=0.8, controller_type=self.controller_type)
        self.interface.angle_vector(self.robot.angle_vector(), time=0.5, controller_type=self.controller_type)

    def joint_command_callback(self, msg: JointState) -> None:
        """
        Get Command from JointState message and send command to robot

        Args:
            msg (JointState): JointState message, which contains Joint angle command
                name (list): Joint names to command
                position (list): Joint angles to command
        """
        joint_names = msg.name
        joint_angles = msg.position

        av = self.robot.angle_vector()
        for joint_name, joint_angle in zip(joint_names, joint_angles):
            joint_id = self.robot.joint_names.index(joint_name)
            av[joint_id] = joint_angle
        self.robot.angle_vector(av)
        self.interface.angle_vector(av)
        self.interface.wait_interpolation()
        self.interface.update_robot_state()
        self.robot.angle_vector(self.interface.potentio_vector())
        print("Joint command executed.")

    def timer_callback(self, event) -> None:
        """
        Timer callback function for updating robot state and publishing controller state
        """

        # Update robot state
        self.interface.update_robot_state()
        self.robot.angle_vector(self.interface.potentio_vector())
        robot_state = self.interface.robot_state

        q = self.arm.angle_vector()
        current_pos = self.arm.end_coords.worldpos()
        current_quat = R.from_matrix(self.arm.end_coords.worldrot()).as_quat()  # xyzw
        effort = []
        for joint_name in self.arm.joint_names:
            joint_id = self.robot.joint_names.index(joint_name)
            effort.append(robot_state["effort"][joint_id])
        effort = np.array(effort)

        header = Header(stamp=rospy.Time.now(), frame_id="world")
        controller_state = ControllerState(
            header=header,
            pose=[
                current_pos[0],
                current_pos[1],
                current_pos[2],
                current_quat[0],
                current_quat[1],
                current_quat[2],
                current_quat[3],
            ],
            q=q,
            effort=effort,
        )

        self.pub_controller_state.publish(controller_state)


if __name__ == "__main__":
    rospy.init_node("pr2_joint_action_controller")
    try:
        controller = PR2JointActionController()
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("pr2_joint_action_controller node terminated.")
