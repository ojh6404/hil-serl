"""Gym Interface for PR2"""
from typing import Dict, Callable
import ast
import os
import numpy as np
import gymnasium as gym
import cv2
import copy
from scipy.spatial.transform import Rotation
import time
import requests
import queue
import threading
from datetime import datetime
from typing import Dict

from serl_robot_infra.pr2_envs.utils.rotations import euler_2_quat, quat_2_euler

class ImageDisplayer(threading.Thread):
    def __init__(self, queue, name):
        threading.Thread.__init__(self)
        self.queue = queue
        self.daemon = True  # make this a daemon thread
        self.name = name

    def run(self):
        while True:
            img_array = self.queue.get()  # retrieve an image from the queue
            if img_array is None:  # None is our signal to exit
                break
            frame = np.concatenate(
                [
                    cv2.resize(v, (128, 128))
                    for k, v in img_array.items()
                    if "full" not in k
                ],
                axis=1,
            )
            cv2.imshow(self.name, frame)
            cv2.waitKey(1)

class DefaultEnvConfig:
    """Default configuration for PR2Env. Fill in the values below."""
    SERVER_URL: str = "http://127.0.0.1:5000/"  # Flask server URL
    CAMERAS: Dict = {
        "kinect_head": "/kinect_head/rgb/image_rect_color/compressed",
    }
    IMAGE_CROP: Dict[str, Callable] = {}
    TARGET_POSE: np.ndarray = np.zeros((6,))            # xyz + rpy
    GRASP_POSE: np.ndarray = np.zeros((6,))             # xyz + rpy
    REWARD_THRESHOLD: np.ndarray = np.zeros((6,))       # xyz + rpy
    ACTION_SCALE: np.ndarray = np.zeros((3,))           # xyz + rpy + gripper (3, )
    RESET_POSE: np.ndarray = np.zeros((6,))             # xyz + rpy
    RANDOM_RESET: bool = False
    RANDOM_XY_RANGE = (0.0,)
    RANDOM_RZ_RANGE = (0.0,)
    ABS_POSE_LIMIT_HIGH: np.ndarray = np.zeros((6,))
    ABS_POSE_LIMIT_LOW: np.ndarray = np.zeros((6,))
    COMPLIANCE_PARAM: Dict[str, float] = {}
    RESET_PARAM: Dict[str, float] = {}
    PRECISION_PARAM: Dict[str, float] = {}
    DISPLAY_IMAGE: bool = True
    GRIPPER_SLEEP: float = 0.6
    MAX_EPISODE_LENGTH: int = 150
    JOINT_RESET_PERIOD: int = 0


class PR2Env(gym.Env):
    def __init__(
        self,
        hz=10,
        fake_env=False,
        save_video=False,
        config: DefaultEnvConfig = None,
    ):
        self.action_scale = config.ACTION_SCALE
        self._TARGET_POSE = config.TARGET_POSE
        self._RESET_POSE = config.RESET_POSE
        self._REWARD_THRESHOLD = config.REWARD_THRESHOLD
        self.url = config.SERVER_URL
        self.config = config
        self.max_episode_length = config.MAX_EPISODE_LENGTH
        self.display_image = config.DISPLAY_IMAGE
        self.gripper_sleep = config.GRIPPER_SLEEP

        self.episode_cnt = 0

        # convert last 3 elements from euler to quat, from size (6,) to (7,)
        self.resetpos = np.concatenate(
            [config.RESET_POSE[:3], euler_2_quat(config.RESET_POSE[3:])]
        )
        self._update_currpos()
        self.last_gripper_act = time.time()
        self.lastsent = time.time()
        self.randomreset = config.RANDOM_RESET
        self.random_xy_range = config.RANDOM_XY_RANGE
        self.random_rz_range = config.RANDOM_RZ_RANGE
        self.hz = hz
        self.joint_reset_cycle = (
            config.JOINT_RESET_PERIOD
        )  # reset the robot joint every 200 cycles

        if save_video:
            print("Saving videos!")
        self.save_video = save_video
        self.recording_frames = []

        # boundary box
        self.xyz_bounding_box = gym.spaces.Box(
            config.ABS_POSE_LIMIT_LOW[:3],
            config.ABS_POSE_LIMIT_HIGH[:3],
            dtype=np.float64,
        )
        self.rpy_bounding_box = gym.spaces.Box(
            config.ABS_POSE_LIMIT_LOW[3:],
            config.ABS_POSE_LIMIT_HIGH[3:],
            dtype=np.float64,
        )
        # Action/Observation Space
        self.action_space = gym.spaces.Box(
            np.ones((7,), dtype=np.float32) * -1,
            np.ones((7,), dtype=np.float32),
        )

        self.observation_space = gym.spaces.Dict(
            {
                "state": gym.spaces.Dict(
                    {
                        "tcp_pose": gym.spaces.Box(
                            -np.inf, np.inf, shape=(7,)
                        ),  # xyz + quat
                        # "tcp_vel": gym.spaces.Box(-np.inf, np.inf, shape=(6,)),
                        "gripper_pose": gym.spaces.Box(-1, 1, shape=(1,)),
                        # "tcp_force": gym.spaces.Box(-np.inf, np.inf, shape=(3,)),
                        # "tcp_torque": gym.spaces.Box(-np.inf, np.inf, shape=(3,)),
                    }
                ),
                "images": gym.spaces.Dict(
                    {
                        key: gym.spaces.Box(0, 255, shape=(128, 128, 3), dtype=np.uint8)
                        for key in config.CAMERAS
                    }
                ),
            }
        )
        self.cycle_count = 0

        if fake_env:
            return

        self.full_images = {}  # New dictionary to store full resolution images
        # Only initialize display if we have cameras
        if len(config.CAMERAS) > 0:
            if self.display_image:
                self.img_queue = queue.Queue()
                self.displayer = ImageDisplayer(self.img_queue, self.url)
                self.displayer.start()
            else:
                from pyvirtualdisplay import Display
                self.display = Display(visible=0, size=(1400, 900))
                self.display.start()

        # Only initialize keyboard listener if we have a display
        if not fake_env and len(config.CAMERAS) > 0:
            try:
                from pynput import keyboard
                self.terminate = False
                def on_press(key):
                    if key == keyboard.Key.esc:
                        self.terminate = True
                self.listener = keyboard.Listener(on_press=on_press)
                self.listener.start()
            except:
                # No display available, skip keyboard listener
                self.terminate = False

        print("Initialized PR2")

    def clip_safety_box(self, pose: np.ndarray) -> np.ndarray:
        """Clip the pose to be within the safety box."""
        pose[:3] = np.clip(
            pose[:3], self.xyz_bounding_box.low, self.xyz_bounding_box.high
        )
        euler = quat_2_euler(pose[3:])

        # Clip first euler angle separately due to discontinuity from pi to -pi
        sign = np.sign(euler[0])
        euler[0] = sign * (
            np.clip(
                np.abs(euler[0]),
                self.rpy_bounding_box.low[0],
                self.rpy_bounding_box.high[0],
            )
        )

        euler[1:] = np.clip(
            euler[1:], self.rpy_bounding_box.low[1:], self.rpy_bounding_box.high[1:]
        )
        pose[3:] = Rotation.from_euler("xyz", euler).as_quat()

        return pose

    def step(self, action: np.ndarray) -> tuple:
        """standard gym step function."""
        # Action : [xyz_delta, rpy_delta, gripper_action]
        start_time = time.time()

        # Clip the action to be within the action space
        action = np.clip(action, self.action_space.low, self.action_space.high) # [-1, 1]
        xyz_delta = action[:3]
        rpy_delta = action[3:6]


        # Update the next position using xyz_delta and rpy_delta
        self.nextpos = self.currpos.copy()
        self.nextpos[:3] = self.nextpos[:3] + xyz_delta * self.action_scale[0]
        self.nextpos[3:] = (
            Rotation.from_euler("xyz", rpy_delta * self.action_scale[1])
            * Rotation.from_quat(self.currpos[3:])
        ).as_quat()
        gripper_action = action[-1] * self.action_scale[2]

        # Send the action to the robot
        self.nextpos = self.clip_safety_box(self.nextpos)
        self._send_gripper_command(gripper_action)
        self._send_pos_command(self.nextpos)

        self.curr_path_length += 1
        dt = time.time() - start_time
        time.sleep(max(0, (1.0 / self.hz) - dt))

        self._update_currpos()
        ob = self._get_obs()
        reward = self.compute_reward(ob)
        done = (
            self.curr_path_length >= self.max_episode_length or self.terminate
        )
        return ob, int(reward), done, False, {"succeed": reward}

    def compute_reward(self, obs) -> float:
        """
        Compute the reward for the current state.
        Dense reward based on distance to target pose, normalized to [0, 1].
        Closer to target = higher reward (max 1.0 when within threshold).
        """

        current_pose = obs["state"]["tcp_pose"]
        print("current_pose:", current_pose)
        # convert from quat to euler first
        current_rot = Rotation.from_quat(current_pose[3:]).as_matrix()
        target_rot = Rotation.from_euler("xyz", self._TARGET_POSE[3:]).as_matrix()
        diff_rot = current_rot.T @ target_rot
        diff_euler = Rotation.from_matrix(diff_rot).as_euler("xyz")
        delta = np.abs(
            np.hstack([current_pose[:3] - self._TARGET_POSE[:3], diff_euler])
        )

        # Normalize delta by threshold (delta/threshold = 0 means at target, >1 means beyond threshold)
        normalized_delta = delta / self._REWARD_THRESHOLD

        # Calculate distance: sqrt of sum of squared normalized deltas
        distance = np.linalg.norm(normalized_delta)

        # Convert to reward: reward = exp(-distance)
        # At target (distance=0): reward=1.0
        # Far from target: reward→0
        reward = np.exp(-distance)

        return float(reward)

    def get_im(self) -> Dict[str, np.ndarray]:
        """
        Get images from camera via flask server.
        sensor data -> ROS -> image data -> flask server -> gym env
        """
        images = {}
        display_images = {}
        full_res_images = {}  # New dictionary to store full resolution cropped images

        for key in self.config.CAMERAS.keys():
            ps = requests.post(self.url + "getimage")
            if ps.status_code == 200:
                shape = ast.literal_eval(ps.headers["X-Image-Shape"])
                dtype = np.dtype(ps.headers["X-Image-Dtype"])
                bgr = np.frombuffer(ps.content, dtype=dtype).reshape(shape)
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                self.full_images[key] = rgb.copy()  # Store the full resolution image
                cropped_rgb = (
                    self.config.IMAGE_CROP[key](rgb)
                    if key in self.config.IMAGE_CROP
                    else rgb
                )
                resized = cv2.resize(
                    cropped_rgb, self.observation_space["images"][key].shape[:2][::-1]
                )
                images[key] = resized[..., ::-1]
                display_images[key] = resized
                display_images[key + "_full"] = cropped_rgb
                full_res_images[key] = copy.deepcopy(
                    cropped_rgb
                )  # Store the full resolution cropped image
            else:
                input(f"Cannot get image from {key}, press enter to re-connect...")
                return self.get_im()

        # Store full resolution cropped images separately
        self.recording_frames.append(full_res_images)

        if self.display_image:
            self.img_queue.put(display_images)
        return images


    def interpolate_move(self, goal: np.ndarray, timeout: float):
        """Move the robot to the goal position with linear interpolation."""
        if goal.shape == (6,):
            goal = np.concatenate([goal[:3], euler_2_quat(goal[3:])])
        steps = int(timeout * self.hz)
        self._update_currpos()
        path = np.linspace(self.currpos, goal, steps)
        for p in path:
            self._send_pos_command(p)
            time.sleep(1 / self.hz)
        self.nextpos = p
        self._update_currpos()

    def go_to_reset(self, joint_reset=True):
        """
        The concrete steps to perform reset should be
        implemented each subclass for the specific task.
        Should override this method if custom reset procedure is needed.
        """
        # Reset and open gripper
        requests.post(self.url + "activate_gripper")
        time.sleep(0.1)
        requests.post(self.url + "open_gripper")
        time.sleep(0.5)

        # Change to precision mode for reset        # Use compliance mode for coupled reset
        self._update_currpos()
        self._send_pos_command(self.currpos)
        time.sleep(0.3)
        requests.post(self.url + "update_param", json=self.config.PRECISION_PARAM)
        time.sleep(0.5)

        # Perform joint reset if needed
        if joint_reset:
            requests.post(self.url + "jointreset")
            time.sleep(1.0)

        # Perform Carteasian reset
        if self.randomreset:  # randomize reset position in xy plane
            reset_pose = self.resetpos.copy()
            reset_pose[:2] += np.random.uniform(
                -self.random_xy_range, self.random_xy_range, (2,)
            )
            euler_random = self._RESET_POSE[3:].copy()
            euler_random[-1] += np.random.uniform(
                -self.random_rz_range, self.random_rz_range
            )
            reset_pose[3:] = euler_2_quat(euler_random)
            self.interpolate_move(reset_pose, timeout=3)
        else:
            reset_pose = self.resetpos.copy()
            self.interpolate_move(reset_pose, timeout=3)

        # Change to compliance mode
        requests.post(self.url + "update_param", json=self.config.COMPLIANCE_PARAM)

    def reset(self, joint_reset=True, **kwargs):
        # print(f"Resetting environment for episode {self.episode_cnt}")
        self.episode_cnt += 1
        self.last_gripper_act = time.time()
        requests.post(self.url + "update_param", json=self.config.COMPLIANCE_PARAM)
        if self.save_video:
            self.save_video_recording()

        self.cycle_count += 1
        if (
            self.joint_reset_cycle != 0
            and self.cycle_count % self.joint_reset_cycle == 0
        ):
            self.cycle_count = 0
            joint_reset = True

        self._recover()
        self.go_to_reset(joint_reset=joint_reset)
        self._recover()
        self.curr_path_length = 0

        self._update_currpos()
        obs = self._get_obs()
        self.terminate = False
        return obs, {"succeed": False}

    def save_video_recording(self):
        try:
            if len(self.recording_frames):
                if not os.path.exists("./videos"):
                    os.makedirs("./videos")

                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

                for camera_key in self.recording_frames[0].keys():
                    if self.url == "http://127.0.0.1:5000/":
                        video_path = f"./videos/left_{camera_key}_{timestamp}.mp4"
                    else:
                        video_path = f"./videos/right_{camera_key}_{timestamp}.mp4"

                    # Get the shape of the first frame for this camera
                    first_frame = self.recording_frames[0][camera_key]
                    height, width = first_frame.shape[:2]

                    video_writer = cv2.VideoWriter(
                        video_path,
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        10,
                        (width, height),
                    )

                    for frame_dict in self.recording_frames:
                        video_writer.write(cv2.cvtColor(frame_dict[camera_key], cv2.COLOR_RGB2BGR))

                    video_writer.release()
                    print(f"Saved video for camera {camera_key} at {video_path}")

            self.recording_frames.clear()
        except Exception as e:
            print(f"Failed to save video: {e}")

    def _recover(self):
        """Internal function to recover the robot from error state."""
        requests.post(self.url + "clearerr")

    def _send_pos_command(self, pos: np.ndarray):
        """Internal function to send position command to the robot."""
        self._recover()
        arr = np.array(pos).astype(np.float32)
        data = {"arr": arr.tolist()}
        requests.post(self.url + "pose", json=data)

    def _send_gripper_command(self, command: float, mode="binary"):
        """Internal function to send gripper command to the robot."""
        if mode == "binary":
            if (
                (command >= 0.5)                   # close gripper
                and (self.curr_gripper_pos > 0.85)  # when gripper is open
                and (time.time() - self.last_gripper_act > self.gripper_sleep)
            ):  # close gripper
                requests.post(self.url + "close_gripper")
                self.last_gripper_act = time.time()
                time.sleep(self.gripper_sleep)
            elif (
                (command <= -0.5)                    # open gripper
                and (self.curr_gripper_pos < 0.85)  # when gripper is closed
                and (time.time() - self.last_gripper_act > self.gripper_sleep)
            ):  # open gripper
                requests.post(self.url + "open_gripper")
                self.last_gripper_act = time.time()
                time.sleep(self.gripper_sleep)
            else:
                return
        elif mode == "continuous":
            raise NotImplementedError("Continuous gripper control is optional")


    def _update_currpos(self):
        """
        Internal function to get the latest state of the robot and its gripper.
        """
        ps = requests.post(self.url + "getstate").json()
        self.currpos = np.array(ps["pose"])
        self.currvel = np.array(ps["vel"])

        self.currforce = np.array(ps["force"])[:3]
        self.currtorque = np.array(ps["force"])[3:]
        self.curreffort = np.array(ps["torque"]) # NOTE not used now
        self.currjacobian = np.reshape(np.array(ps["jacobian"]), (6, 7))
        self.q = np.array(ps["q"])
        self.dq = np.array(ps["dq"])
        self.curr_gripper_pos = np.array(ps["gripper_pos"])

    def _get_obs(self) -> dict:
        images = self.get_im()
        state_observation = {
            "tcp_pose": self.currpos,               # xyz + quat    (7,)
            # "tcp_vel": self.currvel,                # xyz + rpy     (6,)
            "gripper_pose": self.curr_gripper_pos,  # scalar        (1,)
            # "tcp_force": self.currforce,            # xyz           (3,)
            # "tcp_torque": self.currtorque,          # xyz           (3,)
        }
        return copy.deepcopy(dict(images=images, state=state_observation))

    def close(self):
        if hasattr(self, "listener"):
            self.listener.stop()
        if hasattr(self, "img_queue") and self.display_image:
            self.img_queue.put(None)
            cv2.destroyAllWindows()
            self.displayer.join()
        elif hasattr(self, "display"):
            self.display.stop()


if __name__ == "__main__":
    from serl_robot_infra.pr2_envs.envs.wrappers import ROSSpacemouseIntervention, AffordanceWrapper, Quat2EulerWrapper, SparseVLMRewardWrapper
    from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper
    from serl_launcher.wrappers.chunking import ChunkingWrapper

    default_config = DefaultEnvConfig()
    default_config.SERVER_URL = "http://133.11.216.159:5000/"  # Flask server URL
    # default_config.SERVER_URL = "http://127.0.0.1:5000/"  # Flask server URL
    # default_config.TARGET_POSE = np.array([0.80680774,0.1987997,1.02477692, 0, 0, 0.34906585])
    default_config.TARGET_POSE = np.array([0.72043526  0.10875295  1.25475795, 0.0, 0.0, 0.0])
    default_config.GRASP_POSE = np.array([0.5857508505445138,-0.22036261105675414,0.2731021902359492, 0, 0, 0])
    default_config.RESET_POSE = default_config.TARGET_POSE + np.array([-0.3, 0, 0, 0, 0, 0])
    default_config.ABS_POSE_LIMIT_LOW = default_config.TARGET_POSE + np.array([-0.5, -0.4, -0.2, -0.01, -0.1, -0.2])
    default_config.ABS_POSE_LIMIT_HIGH = default_config.TARGET_POSE + np.array([0.1, 0.1, 0.2, 0.01, 0.1, 1.2])
    default_config.REWARD_THRESHOLD = np.array([0.05, 0.05, 0.05, 1.0, 1.0, 1.0])
    default_config.RANDOM_RESET = True
    default_config.RANDOM_XY_RANGE = 0.1
    default_config.RANDOM_RZ_RANGE = 0.05
    default_config.ACTION_SCALE = (0.05, 0.5, 1) # xyz, rpy, gripper
    default_config.DISPLAY_IMAGE = False
    default_config.GRIPPER_SLEEP = 0.0
    default_config.MAX_EPISODE_LENGTH = 100

    default_config.COMPLIANCE_PARAM = {
        "translational_stiffness": 2000,     # Kp (translation)
        "translational_damping": 30,         # Kd (translation)
        "rotational_stiffness": 50,         # Kp (rotation)
        "rotational_damping": 0,            # Kd (rotation)
        "translational_Ki": 0,              # Ki (translation)
        "rotational_Ki": 0,                 # Ki (rotation)
        "translational_clip_x": 0.3,        # Clip translational movement in x direction
        "translational_clip_y": 0.3,        # Clip translational movement in y direction
        "translational_clip_z": 0.3,        # Clip translational movement in z direction
        "translational_clip_neg_x": 0.3,    # Clip translational movement in negative x direction
        "translational_clip_neg_y": 0.3,    # Clip translational movement in negative y direction
        "translational_clip_neg_z": 0.3,    # Clip translational movement in negative z direction
        "rotational_clip_x": 0.5,          # Cl fsip rotational movement in x direction
        "rotational_clip_y": 0.5,          # Clip rotational movement in y direction
        "rotational_clip_z": 0.1,           # Clip rotational movement in z direction
        "rotational_clip_neg_x": 0.5,      # Clip rotational movement in negative x direction
        "rotational_clip_neg_y": 0.5,      # Clip rotational movement in negative y direction
        "rotational_clip_neg_z": 0.1,       # Clip rotational movement in negative z direction
        "clip_target": False,               # Clip target pose flag
    }

    save_video = True

    # proprio_keys = ["tcp_pose", "tcp_vel", "tcp_force", "tcp_torque", "gripper_pose"] # 6 + 6 + 3 + 3 + 1 = 19
    proprio_keys = ["tcp_pose", "gripper_pose"] # 6 + 6 + 3 + 3 + 1 = 19
    env = PR2Env(config=default_config, save_video=save_video)
    env = ROSSpacemouseIntervention(env=env)
    # env = AffordanceWrapper(env)  # Commented out - affordance not needed
    env = SparseVLMRewardWrapper(
        env,
        prompt="Does fridge in the scene open? Answer yes or no.",
        camera_name="kinect_head",  # PR2's head-mounted Kinect camera
        vlm_server_url="http://133.11.216.111:5001/reward",
        update_interval=0.5,
        reward_scale=1.0,
    )
    env = Quat2EulerWrapper(env)
    env = SERLObsWrapper(env, proprio_keys=proprio_keys)
    env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)

    obs = env.reset(joint_reset=True)

    try:
        while True:
            start_time = time.time()
            # sample random between -1.0 and 1.0
            action = np.random.uniform(-1.0, 1.0, size=env.action_space.shape)
            # action = np.zeros(env.action_space.shape)
            # action[0] = 0.1
            obs, reward, done, terminated, info = env.step(action)
            # print("Task Reward: ", info["task_reward"], info["succeed"])

            # Print VLM reward info if available
            if "vlm_reward" in info:
                print(f"VLM Reward: {info['vlm_reward']}, Answer: {info['vlm_answer']}, Confidence: {info['vlm_confidence']}, Success: {info['succeed']}")

            if done or terminated:
                print("Episode done!")
                env.reset(reset_joints=True)
    except KeyboardInterrupt:
        env.reset(joint_reset=True)
        env.close()

