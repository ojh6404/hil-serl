"""
This file starts a control server running on the real time PC connected to the pr2 robot.
In a screen run `python pr2_server.py`
"""

from typing import List
import time
import subprocess
import numpy as np
from scipy.spatial.transform import Rotation as R
from absl import app, flags
from flask import Flask, request, jsonify, Response

import rospy
import tf2_ros
from std_srvs.srv import Trigger
from geometry_msgs.msg import PoseStamped, Point, Quaternion
from sensor_msgs.msg import JointState
from serl_controllers_msgs.msg import ControllerState
from dynamic_reconfigure.client import Client as ReconfClient
from serl_robot_infra.pr2_servers.pr2_gripper_server_left import (
    PR2GripperServer,
)

from serl_robot_infra.pr2_servers.spacenav_server import SpacenavServer
from serl_robot_infra.pr2_servers.camera_server import CameraServer
from serl_robot_infra.pr2_servers.affordance_server import AffordanceServer

R_ARM_JOINT_NAMES = [
    "r_shoulder_pan_joint",
    "r_shoulder_lift_joint",
    "r_upper_arm_roll_joint",
    "r_elbow_flex_joint",
    "r_forearm_roll_joint",
    "r_wrist_flex_joint",
    "r_wrist_roll_joint",
]
L_ARM_JOINT_NAMES = [
    "l_shoulder_pan_joint",
    "l_shoulder_lift_joint",
    "l_upper_arm_roll_joint",
    "l_elbow_flex_joint",
    "l_forearm_roll_joint",
    "l_wrist_flex_joint",
    "l_wrist_roll_joint",
]
TORSO_JOINT_NAMES = ["torso_lift_joint"]
HEAD_JOINT_NAMES = ["head_pan_joint", "head_tilt_joint"]
ACTUATED_JOINT_NAMES = (
    TORSO_JOINT_NAMES + L_ARM_JOINT_NAMES + R_ARM_JOINT_NAMES + HEAD_JOINT_NAMES
)
RESET_JOINT_POSITIONS = {
    "torso_lift_joint": 0.25,  # [m]
    "l_shoulder_pan_joint": np.deg2rad(75),
    "l_shoulder_lift_joint": np.deg2rad(50),
    "l_upper_arm_roll_joint": np.deg2rad(110),
    "l_elbow_flex_joint": np.deg2rad(-110),
    "l_forearm_roll_joint": np.deg2rad(-20),
    "l_wrist_flex_joint": np.deg2rad(-10),
    "l_wrist_roll_joint": np.deg2rad(-10),
    "r_shoulder_pan_joint": np.deg2rad(-75),
    "r_shoulder_lift_joint": np.deg2rad(50),
    "r_upper_arm_roll_joint": np.deg2rad(-110),
    "r_elbow_flex_joint": np.deg2rad(-110),
    "r_forearm_roll_joint": np.deg2rad(20),
    "r_wrist_flex_joint": np.deg2rad(-10),
    "r_wrist_roll_joint": np.deg2rad(-10),
    "head_pan_joint": 0,
    "head_tilt_joint": np.deg2rad(40),
}
R_ARM_RESET_JOINT_POSITIONS = [
    0.5507248819287145,
    0.5945331195784691,
    0.04670280679847538,
    -2.0512294164319353,
    2.6167708252959674,
    -1.5379466060381415,
    -0.029312102907083304,
]
# replace RESET_JOINT_POSITIONS with R_ARM_RESET_JOINT_POSITIONS
# RESET_JOINT_POSITIONS.update(
#     {joint: R_ARM_RESET_JOINT_POSITIONS[i] for i, joint in enumerate(R_ARM_JOINT_NAMES)}
# )

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "robot_ip", "pr1040", "IP address of the pr2 robot's controller box"
)
flags.DEFINE_string("flask_url", "127.0.0.1", "URL for the flask server to run on.")
flags.DEFINE_list(
    "reset_joint_target",
    [RESET_JOINT_POSITIONS[joint] for joint in ACTUATED_JOINT_NAMES],
    "Target joint angles for the robot to reset to",
)


class PR2Server(object):
    """Handles the starting and stopping of the impedance controller
    (as well as backup) joint recovery policy."""

    def __init__(self, robot_ip:str, ros_pkg_name:str, reset_joint_target: List[float]):
        self.robot_ip = robot_ip
        self.ros_pkg_name = ros_pkg_name
        self.reset_joint_target = reset_joint_target

        self.control_mode = "joint_action"

        self.joint_controller = subprocess.Popen(
            [
                "roslaunch",
                self.ros_pkg_name,
                "joint.launch",
            ],
            stdout=subprocess.PIPE,
        )
        time.sleep(3)


        # robot reset service
        self.start_impedance_service = rospy.ServiceProxy("/start_impedance", Trigger)
        self.stop_impedance_service = rospy.ServiceProxy("/stop_impedance", Trigger)
        self.enable_robot_service = rospy.ServiceProxy("/enable_robot", Trigger)
        self.disable_robot_service = rospy.ServiceProxy("/disable_robot", Trigger)

        # Publisher for impedance controller (only for pose)
        self.pub_impendance_pose_command = rospy.Publisher(
            "/l_arm_cartesian_impedance_controller/pose_command",
            PoseStamped,
            queue_size=10,
        )

        # Publisher for joint action controller
        self.pub_joint_action_pose_command = rospy.Publisher(
            "/joint_action_controller/pose_command",
            PoseStamped,
            queue_size=10,
        )
        self.pub_joint_action_joint_command = rospy.Publisher(
            "/joint_action_controller/joint_command",
            JointState,
            queue_size=10,
        )

        # Subscriber for robot state
        self.sub_impedance_controller_state = rospy.Subscriber(
            "/l_arm_cartesian_impedance_controller/controller_state",
            ControllerState,
            self._set_robot_state,
        )
        # self.sub_joint_controller_state = rospy.Subscriber(
        #     "/joint_action_controller/joint_controller_state",
        #     ControllerState,
        #     self._set_robot_state,
        # )
        self.sub_joint_state = rospy.Subscriber(
            "/joint_states",
            JointState,
            self._set_joint_state,
        )

        # tf listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.timer = rospy.Timer(rospy.Duration(1.0 / 30.0), self._timer_callback)

        time.sleep(1)

    def _timer_callback(self, event):
        """
        Timer callback for updating the robot state
        """
        try:
            # get base_footprint to torso_lift_link transform
            # cause in impedance mode, pose command's frame is torso_lift_link
            # but in joint action mode, pose command's frame is base_footprint
            transform = self.tf_buffer.lookup_transform(
                "base_footprint", "torso_lift_link", rospy.Time()
            )
            # to numpy array [x, y, z, qx, qy, qz, qw]
            self.base2torso = np.array(
                [
                    transform.transform.translation.x,
                    transform.transform.translation.y,
                    transform.transform.translation.z,
                    transform.transform.rotation.x,
                    transform.transform.rotation.y,
                    transform.transform.rotation.z,
                    transform.transform.rotation.w,
                ]
            ) # [x, y, z, 0, 0, 0, 1] cause torso_lift_link has no rotation w.r.t base_footprint
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            rospy.logerr("Failed to get transform from base_footprint to torso_lift_link")

    def start_impedance(self):
        """Launches the impedance controller"""
        self.start_impedance_service()
        self.control_mode = "impedance"
        rospy.loginfo("Starting Impedance Controller")
        time.sleep(1)

    def stop_impedance(self):
        """Stops the impedance controller"""
        self.stop_impedance_service()
        self.control_mode = "joint_action"
        rospy.loginfo("Stopping Impedance Controller")
        time.sleep(1)

    def enable_robot(self):
        """Enables the robot"""
        self.enable_robot_service()
        rospy.loginfo("Enabling Robot")
        time.sleep(1)

    def disable_robot(self):
        """Disables the robot"""
        self.disable_robot_service()
        rospy.loginfo("Disabling Robot")
        time.sleep(1)

    def clear(self):
        """Clears any errors"""
        pass

    def _set_joint_state(self, msg):
        self.joint_names = msg.name
        self.joint_positions = msg.position
        self.joint_velocities = msg.velocity
        self.joint_efforts = msg.effort

    def reset_joint(self):
        """Resets Joints (needed after running for hours)"""
        # First Stop impedance
        try:
            self.stop_impedance()
            self.clear()
        except:
            rospy.loginfo("impedance Not Running")
        self.clear()

        # self.joint_controller = subprocess.Popen(
        #     [
        #         "roslaunch",
        #         self.ros_pkg_name,
        #         "joint.launch",
        #     ],
        #     stdout=subprocess.PIPE,
        # )
        # time.sleep(3)

        reset_joint_msg = JointState()
        reset_joint_msg.header.stamp = rospy.Time.now()
        reset_joint_msg.header.frame_id = "base_footprint"
        reset_joint_msg.name = ACTUATED_JOINT_NAMES
        reset_joint_msg.position = self.reset_joint_target
        rospy.loginfo("RUNNING JOINT RESET")
        self.clear()
        time.sleep(1)

        # Wait until target joint angles are reached
        count = 0
        joint_indices = [self.joint_names.index(joint) for joint in ACTUATED_JOINT_NAMES]
        success = True
        while not np.allclose(
            np.array(self.reset_joint_target)
            - np.array([self.joint_positions[i] for i in joint_indices]),
            0,
            atol=1e-3,
            rtol=1e-3,
        ):
            self.pub_joint_action_joint_command.publish(reset_joint_msg)
            time.sleep(0.1)
            count += 1
            if count > 30:
                success = False
                break

        # Stop joint controller
        if success:
            rospy.loginfo("RESET DONE")
        else:
            rospy.loginfo("RESET TIMED OUT")

        time.sleep(1)
        self.clear()
        rospy.loginfo("TERMINATE JOINT RESET")

        # Restart impedece controller
        self.start_impedance()
        rospy.loginfo("impedance STARTED")

    def send_joint_command(self, q: List[float]):
        """
        Moves Joints to a position: q

        Args:
            pose (List[float]): (N,)
        """
        assert len(q) == len(ACTUATED_JOINT_NAMES), f"q should be of length {len(ACTUATED_JOINT_NAMES)}"
        msg = JointState()
        msg.header.stamp = rospy.Time.now()
        msg.name = ACTUATED_JOINT_NAMES
        msg.position = q
        self.pub_joint_action_joint_command.publish(msg)

    def send_pose_command(self, pose: List[float]):
        """
        Moves End Effector to a pose: [x, y, z, qx, qy, qz, qw]

        Args:
            pose (List[float]): [x, y, z, qx, qy, qz, qw] in base_footprint frame
        """
        assert len(pose) == 7, "pose should be of length 7, [x, y, z, qx, qy, qz, qw]"
        msg = PoseStamped()
        msg.header.stamp = rospy.Time.now()
        msg.pose.position = Point(pose[0], pose[1], pose[2])
        msg.pose.orientation = Quaternion(pose[3], pose[4], pose[5], pose[6])
        if self.control_mode == "joint_action":
            # pose command's frame is base_footprint
            msg.header.frame_id = "base_footprint"
            self.pub_joint_action_pose_command.publish(msg)
        elif self.control_mode == "impedance":
            # origin of catersian impedance controller is torso_lift_link, so convert to torso_lift_link frame from base_footprint
            msg.header.frame_id = "torso_lift_link"
            msg.pose.position.x -= self.base2torso[0]
            msg.pose.position.y -= self.base2torso[1]
            msg.pose.position.z -= self.base2torso[2]
            self.pub_impendance_pose_command.publish(msg)
        else:
            rospy.logerr("Invalid control mode")

    def _set_robot_state(self, msg: ControllerState):
        """
        Set Robot State

        Args:
            msg (ControllerState): ControllerState message
                pose (list): End-effector pose [x, y, z, qx, qy, qz, qw]
                vel (list): End-effector velocity [vx, vy, vz, wx, wy, wz]
                q (list): Joint angles [q1, q2, q3, q4, q5, q6, q7]
                qvel (list): Joint velocities [dq1, dq2, dq3, dq4, dq5, dq6, dq7]
                gravity (list): Gravity [fx, fy, fz, tx, ty, tz]
                force (list): Force [fx, fy, fz, tx, ty, tz]
                torque (list): Torque [tx, ty, tz, fx, fy, fz]
                command_torque (list): Command Torque [tx, ty, tz, fx, fy, fz]
                tau_task (list): Task Torque [tx, ty, tz, fx, fy, fz]
                tau_nullspace (list): Nullspace Torque [tx, ty, tz, fx, fy, fz]
                tau_i (list): Inertia Torque [tx, ty, tz, fx, fy, fz]
                jacobian (list): Jacobian, (6, 7)

        """
        self.pos = np.array(list(msg.pose))     # [x, y, z, qx, qy, qz, qw]
        self.q = np.array(list(msg.q))  # [q1, q2, q3, q4, q5, q6, q7]
        self.torque = np.array(list(msg.effort))
        if self.control_mode == "impedance":    # pos's frame is torso_lift_link, so convert to base_footprint
            self.pos[:3] = self.pos[:3] + self.base2torso[:3]
            self.vel = np.array(list(msg.vel))
            self.dq = np.array(list(msg.qvel))  # [dq1, dq2, dq3, dq4, dq5, dq6, dq7]
            self.gravity = np.array(list(msg.gravity))
            self.force = np.array(list(msg.force))
            self.command_torque = np.array(list(msg.command_effort))
            self.tau_task = np.array(list(msg.tau_task))
            self.tau_nullspace = np.array(list(msg.tau_nullspace))
            self.tau_i = np.array(list(msg.tau_i))
            self.jacobian = np.array(list(msg.jacobian)).reshape((6, 7), order="F")



    def __del__(self):
        self.joint_controller.terminate()
        rospy.signal_shutdown("Shutting down")
        time.sleep(1)


###############################################################################


def main(_):
    ROBOT_IP = FLAGS.robot_ip
    ROS_PKG_NAME = "serl_pr2_controllers"
    RESET_JOINT_TARGET = FLAGS.reset_joint_target

    webapp = Flask(__name__)

    # Start ros node
    rospy.init_node("pr2_control_api")

    """Starts impedance controller"""
    robot_server = PR2Server(
        robot_ip=ROBOT_IP,
        ros_pkg_name=ROS_PKG_NAME,
        reset_joint_target=RESET_JOINT_TARGET,
    )
    gripper_server = PR2GripperServer()
    camera_server = CameraServer()
    spacenav_server = SpacenavServer()
    affordance_server = AffordanceServer()

    # Start impedance controller
    robot_server.start_impedance()
    reconf_client = ReconfClient(
        "l_arm_cartesian_impedance_controllerdynamic_reconfigure_compliance_param_node"
    )
    robot_server.enable_robot()

    # Initialize gripper
    gripper_server.reset_gripper()

    # Route for Setting Load
    @webapp.route("/reset", methods=["POST"])
    def reset():
        robot_server.reset_service()
        rospy.loginfo("Reset Robot")
        return "Set Load"

    # Route for Starting impedance
    @webapp.route("/startimp", methods=["POST"])
    def start_impedance():
        robot_server.clear()
        robot_server.start_impedance()
        return "Started impedance"

    # Route for Stopping impedance
    @webapp.route("/stopimp", methods=["POST"])
    def stop_impedance():
        robot_server.stop_impedance()
        return "Stopped impedance"

    # Route for Enabling Robot
    @webapp.route("/enablerobot", methods=["POST"])
    def enable_robot():
        robot_server.enable_robot()
        return "Enabled Robot"

    # Route for Disabling Robot
    @webapp.route("/disablerobot", methods=["POST"])
    def disable_robot():
        robot_server.disable_robot()
        return "Disabled Robot"

    # Route for pose in euler angles
    @webapp.route("/getpos_euler", methods=["POST"])
    def get_pose_euler():
        xyz = robot_server.pos[:3]
        r = R.from_quat(robot_server.pos[3:]).as_euler("xyz")
        return jsonify({"pose": np.concatenate([xyz, r]).tolist()})

    # Route for Getting Pose
    @webapp.route("/getpos", methods=["POST"])
    def get_pos():
        return jsonify({"pose": np.array(robot_server.pos).tolist()})

    @webapp.route("/getvel", methods=["POST"])
    def get_vel():
        return jsonify({"vel": np.array(robot_server.vel).tolist()})

    @webapp.route("/getforce", methods=["POST"])
    def get_force():
        return jsonify({"force": np.array(robot_server.force).tolist()})

    @webapp.route("/gettorque", methods=["POST"])
    def get_torque():
        return jsonify({"torque": np.array(robot_server.torque).tolist()})

    @webapp.route("/getq", methods=["POST"])
    def get_q():
        return jsonify({"q": np.array(robot_server.q).tolist()})

    @webapp.route("/getdq", methods=["POST"])
    def get_dq():
        return jsonify({"dq": np.array(robot_server.dq).tolist()})

    @webapp.route("/getjacobian", methods=["POST"])
    def get_jacobian():
        return jsonify({"jacobian": np.array(robot_server.jacobian).tolist()})

    # Route for getting gripper distance
    @webapp.route("/get_gripper", methods=["POST"])
    def get_gripper():
        return jsonify({"gripper": gripper_server.gripper_pos})

    # Route for Running Joint Reset
    @webapp.route("/jointreset", methods=["POST"])
    def joint_reset():
        robot_server.clear()
        robot_server.reset_joint()
        return "Reset Joint"

    # Route for Activating the Gripper
    @webapp.route("/activate_gripper", methods=["POST"])
    def activate_gripper():
        print("activate gripper")
        return "Activated"

    # Route for Resetting the Gripper. It will reset and activate the gripper
    @webapp.route("/reset_gripper", methods=["POST"])
    def reset_gripper():
        gripper_server.reset_gripper()
        return "Reset"

    # Route for Opening the Gripper
    @webapp.route("/open_gripper", methods=["POST"])
    def open():
        gripper_server.open()
        return "Opened"

    # Route for Closing the Gripper
    @webapp.route("/close_gripper", methods=["POST"])
    def close():
        gripper_server.close()
        return "Closed"

    # Route for Closing the Gripper
    @webapp.route("/close_gripper_slow", methods=["POST"])
    def close_slow():
        gripper_server.close_slow()
        return "Closed"

    # Route for moving the gripper
    @webapp.route("/move_gripper", methods=["POST"])
    def move_gripper():
        gripper_pos = request.json
        pos = np.clip(gripper_pos, 0, 1)
        gripper_server.move_gripper(pos)
        return "Moved Gripper"

    # Route for Clearing Errors (Communcation constraints, etc.)
    @webapp.route("/clearerr", methods=["POST"])
    def clear():
        robot_server.clear()
        return "Clear"

    # Route for Sending a pose command
    @webapp.route("/pose", methods=["POST"])
    def pose():
        pos = np.array(request.json["arr"])
        robot_server.send_pose_command(pos)
        return "Pose Command Sent"

    # Route for Sending a joint command
    @webapp.route("/joint", methods=["POST"])
    def joint():
        joint = np.array(request.json["arr"])
        robot_server.send_joint_command(joint)
        return "Joint Command Sent"

    # Route for getting all state information
    @webapp.route("/getstate", methods=["POST"])
    def get_state():
        robot_state = {
            "pose": np.array(robot_server.pos).tolist(),
            "vel": np.array(robot_server.vel).tolist(),
            "force": np.array(robot_server.force).tolist(),
            "torque": np.array(robot_server.torque).tolist(),
            "q": np.array(robot_server.q).tolist(),
            "dq": np.array(robot_server.dq).tolist(),
            "jacobian": np.array(robot_server.jacobian).tolist(),
            "gripper_pos": gripper_server.gripper_pos,
        } if robot_server.control_mode == "impedance" else {
            "pose": np.array(robot_server.pos).tolist(),
            "q": np.array(robot_server.q).tolist(),
            "torque": np.array(robot_server.torque).tolist(),
            "gripper_pos": gripper_server.gripper_pos,
        }
        return jsonify(robot_state)

    @webapp.route("/getimage", methods=["POST"])
    def get_image():
        return Response(
            camera_server.image.tobytes(),
            mimetype="application/octet-stream",
            headers={
                "X-Image-Shape": f"{camera_server.image.shape}",
                "X-Image-Dtype": str(camera_server.image.dtype),
            },
        )

    # @webapp.route("/getdepth", methods=["POST"])
    # def get_depth():
    #     return Response(
    #         camera_server.depth.tobytes(),
    #         mimetype="application/octet-stream",
    #         headers={
    #             "X-Image-Shape": f"{camera_server.depth.shape}",
    #             "X-Image-Dtype": str(camera_server.depth.dtype),
    #         },
    #     )

    @webapp.route("/getcamera", methods=["POST"])
    def get_camera():
        return jsonify(
            {
                "width": camera_server.width,
                "height": camera_server.height,
                "K": camera_server.K.tolist(),
            }
        )

    # Route for updating compliance parameters
    @webapp.route("/update_param", methods=["POST"])
    def update_param():
        print("Updating compliance parameters")
        reconf_client.update_configuration(request.json)
        return "Updated compliance parameters"

    # Route for getting spacenav state
    @webapp.route("/getjoy", methods=["POST"])
    def get_joy():
        return jsonify(
            {
                "translation": spacenav_server.translation.tolist(),
                "rotation": spacenav_server.rotation.tolist(),
                "buttons": spacenav_server.buttons,
            }
        )

    # Route for getting spacenav state
    @webapp.route("/getaffordance", methods=["POST"])
    def get_affordance():
        return jsonify(
            {
                "grasp_pose": affordance_server.grasp_pose.tolist(),
                "pre_traj": affordance_server.pre_traj.tolist(),
                "post_traj": affordance_server.post_traj.tolist(),
            }
        )


    print("Starting Flask Server")
    webapp.run(host=FLAGS.flask_url)


if __name__ == "__main__":
    app.run(main)
