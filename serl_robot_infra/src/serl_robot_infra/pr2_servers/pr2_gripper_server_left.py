import rospy
from pr2_controllers_msgs.msg import Pr2GripperCommand, JointControllerState

MAX_GRIPPER_POS = 0.09
MAX_GRIPPER_EFFORT = 1000

class PR2GripperServer:
    def __init__(self):
        super().__init__()
        self.pub_gripper_command = rospy.Publisher(
            "/l_gripper_controller/command",
            Pr2GripperCommand,
            queue_size=10,
        )
        self.sub_gripper_state = rospy.Subscriber(
            "/l_gripper_controller/state",
            JointControllerState,
            self._update_gripper,
        )
        self.binary_gripper_pose = 0

    def _update_gripper(self, msg):
        self.gripper_pos = msg.process_value / MAX_GRIPPER_POS # scale to 0-1

    def move(self, position: float):
        """
        pos: float, gripper position between 0 and 1
        """
        msg = Pr2GripperCommand()
        msg.position = position * MAX_GRIPPER_POS # scale to 0-MAX_GRIPPER_POS
        msg.max_effort = MAX_GRIPPER_EFFORT
        self.pub_gripper_command.publish(msg)

    def open(self):
        if self.binary_gripper_pose == 0:
            return
        self.move(1.0)
        self.binary_gripper_pose = 0

    def close(self):
        if self.binary_gripper_pose == 1:
            return
        self.move(0.0)
        self.binary_gripper_pose = 1

    def close_slow(self):
        # just close the gripper
        self.close()

    def reset_gripper(self):
        # just open the gripper
        self.binary_gripper_pose = 1
        self.open()
