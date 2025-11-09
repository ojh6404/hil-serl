import os
import jax
import jax.numpy as jnp
import numpy as np

from serl_robot_infra.pr2_envs.envs.wrappers import (
    GripperCloseEnv,
    Quat2EulerWrapper,
    SparseVLMRewardWrapper,
    # AffordanceWrapper,
    ROSSpacemouseIntervention,
)
from serl_robot_infra.pr2_envs.envs.pr2_env import DefaultEnvConfig
from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper
from serl_launcher.wrappers.chunking import ChunkingWrapper

from experiments.config import DefaultTrainingConfig
from experiments.open_fridge.wrapper import FridgeEnv

DLBOX14_IP = os.environ.get("DLBOX14_IP", "133.11.216.111")

class EnvConfig(DefaultEnvConfig):
    # IP = "127.0.0.1"
    IP = "133.11.216.159"
    SERVER_URL = f"http://{IP}:5000/"
    CAMERAS = {
        "kinect_head": "/kinect_head/rgb/image_rect_color/compressed",
    }
    IMAGE_CROP = {
        "kinect_head": lambda img: img[80:500, 80:480],
    }
    TARGET_POSE = np.array([0.7386425275196229,0.2571240389863525,1.0385435180520572, 0, 0, 0.1])
    GRASP_POSE = np.array([0.5857508505445138,-0.22036261105675414,0.2731021902359492, 0, 0, 0])
    RESET_POSE = TARGET_POSE + np.array([-0.3, 0, 0, 0, 0, 0])
    ABS_POSE_LIMIT_LOW = TARGET_POSE + np.array([-0.45, -0.4, -0.2, -0.01, -0.1, -0.2])
    ABS_POSE_LIMIT_HIGH = TARGET_POSE + np.array([0.05, 0.1, 0.2, 0.01, 0.1, 1.2])
    REWARD_THRESHOLD = np.array([0.05, 0.05, 0.05, 1.0, 1.0, 1.0])
    RANDOM_RESET = True
    RANDOM_XY_RANGE = 0.1
    RANDOM_RZ_RANGE = 0.05
    ACTION_SCALE = (0.05, 0.3, 1)
    DISPLAY_IMAGE = False
    MAX_EPISODE_LENGTH = 150
    COMPLIANCE_PARAM = {
        "translational_stiffness": 1500,     # Kp (translation)
        "translational_damping": 30,         # Kd (translation)
        "rotational_stiffness": 50,         # Kp (rotation)
        "rotational_damping": 0,            # Kd (rotation)
        "translational_Ki": 0,              # Ki (translation)
        "rotational_Ki": 0,                 # Ki (rotation)
        "translational_clip_x": 0.2,        # Clip translational movement in x direction
        "translational_clip_y": 0.2,        # Clip translational movement in y direction
        "translational_clip_z": 0.2,        # Clip translational movement in z direction
        "translational_clip_neg_x": 0.1,    # Clip translational movement in negative x direction
        "translational_clip_neg_y": 0.1,    # Clip translational movement in negative y direction
        "translational_clip_neg_z": 0.1,    # Clip translational movement in negative z direction
        "rotational_clip_x": 0.01,          # Cl fsip rotational movement in x direction
        "rotational_clip_y": 0.01,          # Clip rotational movement in y direction
        "rotational_clip_z": 0.5,           # Clip rotational movement in z direction
        "rotational_clip_neg_x": 0.01,      # Clip rotational movement in negative x direction
        "rotational_clip_neg_y": 0.01,      # Clip rotational movement in negative y direction
        "rotational_clip_neg_z": 0.5,       # Clip rotational movement in negative z direction
        "clip_target": False,               # Clip target pose flag
    }
    AFTER_GRASP_PARAM = {
        "translational_stiffness": 2000,     # Kp (translation)
        "translational_damping": 50,         # Kd (translation)
        "rotational_stiffness": 50,         # Kp (rotation)
        "rotational_damping": 0,            # Kd (rotation)
        "translational_Ki": 0,              # Ki (translation)
        "rotational_Ki": 0,                 # Ki (rotation)
        "translational_clip_x": 0.3,        # Clip translational movement in x direction
        "translational_clip_y": 0.3,        # Clip translational movement in y direction
        "translational_clip_z": 0.1,        # Clip translational movement in z direction
        "translational_clip_neg_x": 0.3,    # Clip translational movement in negative x direction
        "translational_clip_neg_y": 0.3,    # Clip translational movement in negative y direction
        "translational_clip_neg_z": 0.1,    # Clip translational movement in negative z direction
        "rotational_clip_x": 0.01,          # Cl fsip rotational movement in x direction
        "rotational_clip_y": 0.01,          # Clip rotational movement in y direction
        "rotational_clip_z": 0.5,           # Clip rotational movement in z direction
        "rotational_clip_neg_x": 0.01,      # Clip rotational movement in negative x direction
        "rotational_clip_neg_y": 0.01,      # Clip rotational movement in negative y direction
        "rotational_clip_neg_z": 0.5,       # Clip rotational movement in negative z direction
        "clip_target": False,               # Clip target pose flag
    }
    # PRECISION_PARAM = {
    #     "translational_stiffness": 1000,
    #     "translational_damping": 0,
    #     "rotational_stiffness": 50,
    #     "rotational_damping": 0,
    #     "translational_Ki": 0.0,
    #     "translational_clip_x": 0.1,
    #     "translational_clip_y": 0.1,
    #     "translational_clip_z": 0.1,
    #     "translational_clip_neg_x": 0.1,
    #     "translational_clip_neg_y": 0.1,
    #     "translational_clip_neg_z": 0.1,
    #     "rotational_clip_x": 0.5,
    #     "rotational_clip_y": 0.5,
    #     "rotational_clip_z": 0.5,
    #     "rotational_clip_neg_x": 0.5,
    #     "rotational_clip_neg_y": 0.5,
    #     "rotational_clip_neg_z": 0.5,
    #     "rotational_Ki": 0.0,
    # }


class TrainConfig(DefaultTrainingConfig):
    image_keys = ["kinect_head"]
    classifier_keys = ["kinect_head"]
    # proprio_keys = ["tcp_pose", "tcp_vel", "tcp_force", "tcp_torque", "gripper_pose"]
    proprio_keys = ["tcp_pose", "gripper_pose"]
    buffer_period = 1000
    checkpoint_period = 5000
    steps_per_update = 50
    encoder_type = "resnet-pretrained"
    setup_mode = "single-arm-fixed-gripper" # or single-arm-learned-gripper TODO

    def get_environment(self, fake_env=False, save_video=False, classifier=False):
        env = FridgeEnv(
            fake_env=fake_env,
            save_video=save_video,
            config=EnvConfig(),
        )
        if not fake_env:
            env = SparseVLMRewardWrapper(
                env,
                prompt="Does fridge in the scene open? Answer yes or no.",
                camera_name="kinect_head",  # PR2's head-mounted Kinect camera
                vlm_server_url=f"http://{DLBOX14_IP}:5001/reward",
                update_interval=0.5,
                reward_scale=1.0,
            )
            env = ROSSpacemouseIntervention(env=env)
        # env = AffordanceWrapper(env)
        env = Quat2EulerWrapper(env)
        env = SERLObsWrapper(env, proprio_keys=self.proprio_keys)
        env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)

        return env
