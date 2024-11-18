import os
import jax
import jax.numpy as jnp
import numpy as np

from serl_robot_infra.robots.wrappers import (
    Quat2EulerWrapper,
    SpacemouseIntervention,
    MultiCameraBinaryRewardClassifierWrapper,
    GripperCloseEnv
)
from serl_robot_infra.robots.wrappers.relative_env import RelativeFrame
from serl_robot_infra.robots.pr2.envs.pr2_env import DefaultEnvConfig
from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper
from serl_launcher.wrappers.chunking import ChunkingWrapper
from serl_launcher.networks.reward_classifier import load_classifier_func

from experiments.config import DefaultTrainingConfig
from experiments.open_fridge.wrapper import FridgeEnv

class EnvConfig(DefaultEnvConfig):
    SERVER_URL = "http://127.0.0.1:5000/"
    CAMERAS = {
        "kinect_head": "/kinect_head/rgb/image_rect_color/compressed",
    }
    IMAGE_CROP = {
        "kinect_head": lambda img: img[120:360, 160:480],
    }
    # TARGET_POSE = np.array([0.5848735137878012, 0.04568075980735583, 0.07919851995276968, np.pi, 0, 0])
    TARGET_POSE = np.array([0.7617406823864483, 0.2502782033779364, 0.024677312442079644, np.pi, 0, 0])
    GRASP_POSE = np.array([0.5857508505445138,-0.22036261105675414,0.2731021902359492, np.pi, 0, 0])
    RESET_POSE = TARGET_POSE + np.array([-0.2, 0, 0, 0, 0, 0])
    ABS_POSE_LIMIT_LOW = TARGET_POSE - np.array([0.3, 0.2, 0.1, 0.01, 0.1, 0.4])
    ABS_POSE_LIMIT_HIGH = TARGET_POSE + np.array([0.0, 0.2, 0.1, 0.01, 0.1, 0.4])
    REWARD_THRESHOLD = np.array([0.05, 0.05, 0.05, 1.0, 1.0, 1.0])
    RANDOM_RESET = True
    RANDOM_XY_RANGE = 0.1
    RANDOM_RZ_RANGE = 0.05
    ACTION_SCALE = (0.05, 0.05, 1)
    DISPLAY_IMAGE = False
    MAX_EPISODE_LENGTH = 100
    COMPLIANCE_PARAM = {
        "translational_stiffness": 1000,     # Kp (translation)
        "translational_damping": 30,         # Kd (translation)
        "rotational_stiffness": 50,         # Kp (rotation)
        "rotational_damping": 0,            # Kd (rotation)
        "translational_Ki": 0,              # Ki (translation)
        "rotational_Ki": 0,                 # Ki (rotation)
        "translational_clip_x": 0.1,        # Clip translational movement in x direction
        "translational_clip_y": 0.1,        # Clip translational movement in y direction
        "translational_clip_z": 0.1,        # Clip translational movement in z direction
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
    PRECISION_PARAM = {
        "translational_stiffness": 300,
        "translational_damping": 0,
        "rotational_stiffness": 50,
        "rotational_damping": 0,
        "translational_Ki": 0.0,
        "translational_clip_x": 0.1,
        "translational_clip_y": 0.1,
        "translational_clip_z": 0.1,
        "translational_clip_neg_x": 0.1,
        "translational_clip_neg_y": 0.1,
        "translational_clip_neg_z": 0.1,
        "rotational_clip_x": 0.5,
        "rotational_clip_y": 0.5,
        "rotational_clip_z": 0.5,
        "rotational_clip_neg_x": 0.5,
        "rotational_clip_neg_y": 0.5,
        "rotational_clip_neg_z": 0.5,
        "rotational_Ki": 0.0,
    }


class TrainConfig(DefaultTrainingConfig):
    image_keys = ["kinect_head"]
    classifier_keys = ["kinect_head"]
    proprio_keys = ["tcp_pose", "tcp_vel", "tcp_force", "tcp_torque", "gripper_pose"]
    buffer_period = 1000
    checkpoint_period = 5000
    steps_per_update = 50
    encoder_type = "resnet-pretrained"
    setup_mode = "single-arm-fixed-gripper" # or single-arm-learned-gripper

    def get_environment(self, fake_env=False, save_video=False, classifier=False):
        env = FridgeEnv(
            fake_env=fake_env,
            save_video=save_video,
            config=EnvConfig(),
        )
        # env = GripperCloseEnv(env)
        if not fake_env:
            env = SpacemouseIntervention(env)
        env = RelativeFrame(env)
        env = Quat2EulerWrapper(env)
        env = SERLObsWrapper(env, proprio_keys=self.proprio_keys)
        env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)
        if classifier:
            classifier = load_classifier_func(
                key=jax.random.PRNGKey(0),
                sample=env.observation_space.sample(),
                image_keys=self.classifier_keys,
                checkpoint_path=os.path.abspath("classifier_ckpt/"),
            )

            def reward_func(obs):
                sigmoid = lambda x: 1 / (1 + jnp.exp(-x))
                # added check for z position to further robustify classifier, but should work without as well
                return int(sigmoid(classifier(obs)) > 0.85 and obs['state'][0, 6] > 0.04)

            env = MultiCameraBinaryRewardClassifierWrapper(env, reward_func)
        return env
