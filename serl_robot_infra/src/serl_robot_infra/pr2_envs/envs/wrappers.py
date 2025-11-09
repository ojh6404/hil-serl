from typing import List
import time
from gymnasium import Env, spaces
import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box
import copy
from pr2_envs.spacemouse.spacemouse_expert import SpaceMouseExpert
import requests
from scipy.spatial.transform import Rotation as R
from PIL import Image
from pr2_envs.envs.pr2_env import PR2Env
import requests

from collections import deque

from pr2_envs.utils.server import pil_to_base64
from pr2_envs.utils.rotations import euler_2_quat, quat_2_euler

sigmoid = lambda x: 1 / (1 + np.exp(-x))

class HumanClassifierWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
    
    def step(self, action):
        obs, rew, done, truncated, info = self.env.step(action)
        if done:
            while True:
                try:
                    rew = int(input("Success? (1/0)"))
                    assert rew == 0 or rew == 1
                    break
                except:
                    continue
        info['succeed'] = rew
        return obs, rew, done, truncated, info
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return obs, info
    
class MultiCameraBinaryRewardClassifierWrapper(gym.Wrapper):
    """
    This wrapper uses the camera images to compute the reward,
    which is not part of the observation space
    """

    def __init__(self, env: Env, reward_classifier_func, target_hz = None):
        super().__init__(env)
        self.reward_classifier_func = reward_classifier_func
        self.target_hz = target_hz

    def compute_reward(self, obs):
        if self.reward_classifier_func is not None:
            return self.reward_classifier_func(obs)
        return 0

    def step(self, action):
        start_time = time.time()
        obs, rew, done, truncated, info = self.env.step(action)
        rew = self.compute_reward(obs)
        done = done or rew
        info['succeed'] = bool(rew)
        if self.target_hz is not None:
            time.sleep(max(0, 1/self.target_hz - (time.time() - start_time)))
            
        return obs, rew, done, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        info['succeed'] = False
        return obs, info
    
    
class MultiStageBinaryRewardClassifierWrapper(gym.Wrapper):
    def __init__(self, env: Env, reward_classifier_func: List[callable]):
        super().__init__(env)
        self.reward_classifier_func = reward_classifier_func
        self.received = [False] * len(reward_classifier_func)
    
    def compute_reward(self, obs):
        rewards = [0] * len(self.reward_classifier_func)
        for i, classifier_func in enumerate(self.reward_classifier_func):
            if self.received[i]:
                continue

            logit = classifier_func(obs).item()
            if sigmoid(logit) >= 0.75:
                self.received[i] = True
                rewards[i] = 1

        reward = sum(rewards)
        return reward

    def step(self, action):
        obs, rew, done, truncated, info = self.env.step(action)
        rew = self.compute_reward(obs)
        done = (done or all(self.received)) # either environment done or all rewards satisfied
        info['succeed'] = all(self.received)
        return obs, rew, done, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.received = [False] * len(self.reward_classifier_func)
        info['succeed'] = False
        return obs, info


class Quat2EulerWrapper(gym.ObservationWrapper):
    """
    Convert the quaternion representation of the tcp pose to euler angles
    """

    def __init__(self, env: Env):
        super().__init__(env)
        assert env.observation_space["state"]["tcp_pose"].shape == (7,)
        # from xyz + quat to xyz + euler
        self.observation_space["state"]["tcp_pose"] = spaces.Box(
            -np.inf, np.inf, shape=(6,)
        )

    def observation(self, observation):
        # convert tcp pose from quat to euler
        tcp_pose = observation["state"]["tcp_pose"]
        observation["state"]["tcp_pose"] = np.concatenate(
            (tcp_pose[:3], R.from_quat(tcp_pose[3:]).as_euler("xyz"))
        )
        return observation

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self.observation(obs), info


class Quat2R2Wrapper(gym.ObservationWrapper):
    """
    Convert the quaternion representation of the tcp pose to rotation matrix
    """

    def __init__(self, env: Env):
        super().__init__(env)
        assert env.observation_space["state"]["tcp_pose"].shape == (7,)
        # from xyz + quat to xyz + euler
        self.observation_space["state"]["tcp_pose"] = spaces.Box(
            -np.inf, np.inf, shape=(9,)
        )

    def observation(self, observation):
        tcp_pose = observation["state"]["tcp_pose"]
        r = R.from_quat(tcp_pose[3:]).as_matrix()
        observation["state"]["tcp_pose"] = np.concatenate(
            (tcp_pose[:3], r[..., :2].flatten())
        )
        return observation

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self.observation(obs), info


class DualQuat2EulerWrapper(gym.ObservationWrapper):
    """
    Convert the quaternion representation of the tcp pose to euler angles
    """

    def __init__(self, env: Env):
        super().__init__(env)
        assert env.observation_space["state"]["left/tcp_pose"].shape == (7,)
        assert env.observation_space["state"]["right/tcp_pose"].shape == (7,)
        # from xyz + quat to xyz + euler
        self.observation_space["state"]["left/tcp_pose"] = spaces.Box(
            -np.inf, np.inf, shape=(6,)
        )
        self.observation_space["state"]["right/tcp_pose"] = spaces.Box(
            -np.inf, np.inf, shape=(6,)
        )

    def observation(self, observation):
        # convert tcp pose from quat to euler
        tcp_pose = observation["state"]["left/tcp_pose"]
        observation["state"]["left/tcp_pose"] = np.concatenate(
            (tcp_pose[:3], R.from_quat(tcp_pose[3:]).as_euler("xyz"))
        )
        tcp_pose = observation["state"]["right/tcp_pose"]
        observation["state"]["right/tcp_pose"] = np.concatenate(
            (tcp_pose[:3], R.from_quat(tcp_pose[3:]).as_euler("xyz"))
        )
        return observation
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self.observation(obs), info

class GripperCloseEnv(gym.ActionWrapper):
    """
    Use this wrapper to task that requires the gripper to be closed
    """

    def __init__(self, env):
        super().__init__(env)
        ub = self.env.action_space
        assert ub.shape == (7,)
        self.action_space = Box(ub.low[:6], ub.high[:6])

    def action(self, action: np.ndarray) -> np.ndarray:
        new_action = np.zeros((7,), dtype=np.float32)
        new_action[:6] = action.copy()
        return new_action

    def step(self, action):
        new_action = self.action(action)
        obs, rew, done, truncated, info = self.env.step(new_action)
        if "intervene_action" in info:
            info["intervene_action"] = info["intervene_action"][:6]
        return obs, rew, done, truncated, info
    
    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    
class SpacemouseIntervention(gym.ActionWrapper):
    def __init__(self, env, action_indices=None):
        super().__init__(env)

        self.gripper_enabled = True
        if self.action_space.shape == (6,):
            self.gripper_enabled = False

        self.expert = SpaceMouseExpert()
        self.left, self.right = False, False
        self.action_indices = action_indices

    def action(self, action: np.ndarray) -> np.ndarray:
        """
        Input:
        - action: policy action
        Output:
        - action: spacemouse action if nonezero; else, policy action
        """
        expert_a, buttons = self.expert.get_action()
        self.left, self.right = tuple(buttons)
        intervened = False
        
        if np.linalg.norm(expert_a) > 0.001:
            intervened = True

        if self.gripper_enabled:
            if self.left:  # close gripper
                gripper_action = np.random.uniform(-1, -0.9, size=(1,))
                intervened = True
            elif self.right:  # open gripper
                gripper_action = np.random.uniform(0.9, 1, size=(1,))
                intervened = True
            else:
                gripper_action = np.zeros((1,))
            expert_a = np.concatenate((expert_a, gripper_action), axis=0)
            expert_a[:6] += np.random.uniform(-0.5, 0.5, size=6)

        if self.action_indices is not None:
            filtered_expert_a = np.zeros_like(expert_a)
            filtered_expert_a[self.action_indices] = expert_a[self.action_indices]
            expert_a = filtered_expert_a

        if intervened:
            return expert_a, True

        return action, False

    def step(self, action):

        new_action, replaced = self.action(action)

        obs, rew, done, truncated, info = self.env.step(new_action)
        if replaced:
            info["intervene_action"] = new_action
        info["left"] = self.left
        info["right"] = self.right
        return obs, rew, done, truncated, info

class DualSpacemouseIntervention(gym.ActionWrapper):
    def __init__(self, env, action_indices=None, gripper_enabled=True):
        super().__init__(env)

        self.gripper_enabled = gripper_enabled

        self.expert = SpaceMouseExpert()
        self.left1, self.left2, self.right1, self.right2 = False, False, False, False
        self.action_indices = action_indices

    def action(self, action: np.ndarray) -> np.ndarray:
        """
        Input:
        - action: policy action
        Output:
        - action: spacemouse action if nonezero; else, policy action
        """
        intervened = False
        expert_a, buttons = self.expert.get_action()
        self.left1, self.left2, self.right1, self.right2 = tuple(buttons)


        if self.gripper_enabled:
            if self.left1:  # close gripper
                left_gripper_action = np.random.uniform(-1, -0.9, size=(1,))
                intervened = True
            elif self.left2:  # open gripper
                left_gripper_action = np.random.uniform(0.9, 1, size=(1,))
                intervened = True
            else:
                left_gripper_action = np.zeros((1,))

            if self.right1:  # close gripper
                right_gripper_action = np.random.uniform(-1, -0.9, size=(1,))
                intervened = True
            elif self.right2:  # open gripper
                right_gripper_action = np.random.uniform(0.9, 1, size=(1,))
                intervened = True
            else:
                right_gripper_action = np.zeros((1,))
            expert_a = np.concatenate(
                (expert_a[:6], left_gripper_action, expert_a[6:], right_gripper_action),
                axis=0,
            )

        if self.action_indices is not None:
            filtered_expert_a = np.zeros_like(expert_a)
            filtered_expert_a[self.action_indices] = expert_a[self.action_indices]
            expert_a = filtered_expert_a

        if np.linalg.norm(expert_a) > 0.001:
            intervened = True

        if intervened:
            return expert_a, True
        return action, False

    def step(self, action):

        new_action, replaced = self.action(action)

        obs, rew, done, truncated, info = self.env.step(new_action)
        if replaced:
            info["intervene_action"] = new_action
        info["left1"] = self.left1
        info["left2"] = self.left2
        info["right1"] = self.right1
        info["right2"] = self.right2
        return obs, rew, done, truncated, info
    
    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


class GripperPenaltyWrapper(gym.RewardWrapper):
    def __init__(self, env, penalty=0.1):
        super().__init__(env)
        assert env.action_space.shape == (7,)
        self.penalty = penalty
        self.last_gripper_pos = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.last_gripper_pos = obs["state"][0, 0]
        return obs, info

    def reward(self, reward: float, action) -> float:
        if (action[6] < -0.5 and self.last_gripper_pos > 0.95) or (
            action[6] > 0.5 and self.last_gripper_pos < 0.95
        ):
            return reward - self.penalty
        else:
            return reward

    def step(self, action):
        """Modifies the :attr:`env` :meth:`step` reward using :meth:`self.reward`."""
        observation, reward, terminated, truncated, info = self.env.step(action)
        if "intervene_action" in info:
            action = info["intervene_action"]
        reward = self.reward(reward, action)
        self.last_gripper_pos = observation["state"][0, 0]
        return observation, reward, terminated, truncated, info

class DualGripperPenaltyWrapper(gym.RewardWrapper):
    def __init__(self, env, penalty=0.1):
        super().__init__(env)
        assert env.action_space.shape == (14,)
        self.penalty = penalty
        self.last_gripper_pos_left = 0 #TODO: this assume gripper starts opened
        self.last_gripper_pos_right = 0 #TODO: this assume gripper starts opened
    
    def reward(self, reward: float, action) -> float:
        if (action[6] < -0.5 and self.last_gripper_pos_left==0):
            reward -= self.penalty
            self.last_gripper_pos_left = 1
        elif (action[6] > 0.5 and self.last_gripper_pos_left==1):
            reward -= self.penalty
            self.last_gripper_pos_left = 0
        if (action[13] < -0.5 and self.last_gripper_pos_right==0):
            reward -= self.penalty
            self.last_gripper_pos_right = 1
        elif (action[13] > 0.5 and self.last_gripper_pos_right==1):
            reward -= self.penalty
            self.last_gripper_pos_right = 0
        return reward
    
    def step(self, action):
        """Modifies the :attr:`env` :meth:`step` reward using :meth:`self.reward`."""
        observation, reward, terminated, truncated, info = self.env.step(action)
        if "intervene_action" in info:
            action = info["intervene_action"]
        reward = self.reward(reward, action)
        return observation, reward, terminated, truncated, info


class ROSSpacemouseIntervention(gym.ActionWrapper):
    def __init__(self, env, action_indices=None):
        super().__init__(env)

        self.gripper_enabled = True
        if self.action_space.shape == (6,):
            self.gripper_enabled = False

        self.left, self.right = False, False
        self.action_indices = action_indices

    def action(self, action: np.ndarray) -> np.ndarray:
        """
        Input:
        - action: policy action
        Output:
        - action: spacemouse action if nonezero; else, policy action
        """

        ps = requests.post(self.url + "getjoy").json()

        translation = np.array(ps["translation"]) # x, y, z [-1 ~ 1]
        rotation = np.array(ps["rotation"]) # roll, pitch, yaw [-1 ~ 1]
        buttons = np.array(ps["buttons"]) # left, right [0, 1]
        expert_a = np.concatenate((translation, rotation), axis=0) # x, y, z, roll, pitch, yaw

        # expert_a, buttons = self.expert.get_action()
        self.left, self.right = tuple(buttons.astype(int).tolist())
        intervened = False

        if np.linalg.norm(expert_a) > 0.05: # TODO
            intervened = True

        if self.gripper_enabled:
            if self.left:  # open gripper
                gripper_action = np.array([-1.0])
                # gripper_action = np.random.uniform(-1, -0.9, size=(1,))
                intervened = True
            elif self.right:  # close gripper
                gripper_action = np.array([1.0])
                # gripper_action = np.random.uniform(0.9, 1, size=(1,))
                intervened = True
            else:
                gripper_action = np.zeros((1,))
            expert_a = np.concatenate((expert_a, gripper_action), axis=0)
            # Removed random noise - it was adding unwanted noise to all actions
            # expert_a[:6] += np.random.uniform(-0.5, 0.5, size=6)

        if self.action_indices is not None:
            filtered_expert_a = np.zeros_like(expert_a)
            filtered_expert_a[self.action_indices] = expert_a[self.action_indices]
            expert_a = filtered_expert_a

        if intervened:
            return expert_a, True

        return action, False

    def step(self, action):
        new_action, replaced = self.action(action)

        obs, rew, done, truncated, info = self.env.step(new_action)
        if replaced:
            info["intervene_action"] = new_action
        info["left"] = self.left
        info["right"] = self.right
        return obs, rew, done, truncated, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


class DenseTaskRewardWrapper(gym.RewardWrapper):
    """
    A reward wrapper that adds CLIP-based dense task rewards based on similarity to a positive prompt.
    The wrapper maintains a cached probability value that is continuously updated in the background.
    """

    def __init__(
        self,
        env: gym.Env,
        positive_prompt: str,
        negative_prompt: str,
        camera_name: str,
        clip_server_url: str = "http://localhost:5005/getprob",
        reward_scale: float =1.0,
        update_interval: float = 0.1,  # Time between CLIP probability updates
        history_length: int = 5,
        min_prob: float = 0.3,
        max_prob: float = 0.6,
    ):
        """
        Initialize the TaskRewardWrapper.

        Args:
            env: The environment to wrap
            positive_prompt: Prompt describing the desired state (e.g., "a photo of opened fridge")
            negative_prompt: Prompt describing the undesired state (e.g., "a photo of closed fridge")
            clip_server_url: URL of the CLIP server endpoint
            reward_scale: Scaling factor for the CLIP-based reward
            update_interval: Minimum time (in seconds) between CLIP probability updates
            history_length: Number of previous task rewards to store for EMA
            min_prob: Minimum probability value to use for task reward
            max_prob: Maximum probability value to use for task reward
        """
        super().__init__(env)
        import threading

        self.positive_prompt = positive_prompt
        self.negative_prompt = negative_prompt
        self.camera_name = camera_name
        self.clip_server_url = clip_server_url
        self.reward_scale = reward_scale
        self.update_interval = update_interval
        self.history_length = history_length

        self.min_prob = min_prob
        self.max_prob = max_prob

        # for EMA
        self.task_reward_history = deque(maxlen=self.history_length)


        # Initialize variables
        self.task_reward = 0.0
        self.last_image = None
        self.last_update_time = 0.0

        # Threading setup
        self._stop_thread = False
        self._thread_lock = threading.Lock()
        self._update_thread = threading.Thread(target=self._continuous_update, daemon=True)
        self._update_thread.start()


    def _get_prob(self, image: Image.Image) -> float:
        """
        Get the CLIP probability for the current image matching the positive prompt.

        Args:
            image: PIL Image to evaluate

        Returns:
            float: Probability of the image matching the positive prompt
        """
        try:
            # Convert image to base64
            img_base64 = pil_to_base64(image)

            # Prepare the request data
            data = {
                "img": img_base64,
                "shape": (image.height, image.width, 3),  # (height, width, channels)
                "positive_prompt": self.positive_prompt,
                "negative_prompt": self.negative_prompt
            }

            # Send request to CLIP server
            response = requests.post(self.clip_server_url, json=data)
            response.raise_for_status()

            # Extract probability
            prob = response.json()["prob"]
            return prob

        except Exception as e:
            print(f"Error getting probability: {str(e)}")
            return self.task_reward  # Return last known probability on error

    def _continuous_update(self):
        """Background thread function to continuously update the probability."""
        while not self._stop_thread:
            current_time = time.time()
            if current_time - self.last_update_time >= self.update_interval:
                if self.last_image is not None:
                    try:
                        image = self.last_image.copy()
                        if len(image.shape) == 4: # batched image
                            image = image[0]
                        if isinstance(image, np.ndarray):
                            image = Image.fromarray(image.astype('uint8'), 'RGB')
                        new_prob = self._get_prob(image)
                        with self._thread_lock:
                            self.task_reward = new_prob
                            self.last_update_time = current_time

                    except Exception as e:
                        print(f"Error in continuous update: {str(e)}")
                else:
                    pass


            # Small sleep to prevent excessive CPU usage
            time.sleep(0.01)

    def reward(self, reward: float) -> float:
        with self._thread_lock:
            task_reward = self.task_reward
        self.task_reward_history.append(task_reward)
        task_reward = np.mean(self.task_reward_history)

        # scale the reward using min_prob and max_prob and make it 0 to 1
        # print("task reward: ", task_reward)
        task_reward = (task_reward - self.min_prob) / (self.max_prob - self.min_prob)
        task_reward = np.clip(task_reward, 0, 1)

        total_reward = reward + self.reward_scale * task_reward
        return total_reward, task_reward

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.last_image = obs["images"][self.camera_name].copy()
        return obs, info

    def step(self, action):
        """Modifies the :attr:`env` :meth:`step` reward using :meth:`self.reward`."""
        observation, reward, terminated, truncated, info = self.env.step(action)
        reward, task_reward = self.reward(reward)
        self.last_image = observation["images"][self.camera_name]
        info["task_reward"] = task_reward
        if task_reward > 0.9:
            info["succeed"] = True
        else:
            info["succeed"] = False
        return observation, reward, terminated, truncated, info

    def close(self):
        """Clean up resources when environment is closed."""
        self._stop_thread = True
        if self._update_thread.is_alive():
            self._update_thread.join(timeout=1.0)
        super().close()

class SparseVLMRewardWrapper(gym.RewardWrapper):
    """
    A reward wrapper that uses VLM-based yes/no questions for sparse reward computation.
    The wrapper maintains a cached reward value that is continuously updated in the background
    to handle VLM's slow inference time.
    """

    def __init__(
        self,
        env: gym.Env,
        prompt: str,
        camera_name: str,
        vlm_server_url: str = "http://localhost:5001/reward",
        update_interval: float = 0.5,  # Time between VLM updates (slower than CLIP)
        reward_scale: float = 1.0,
        temperature: float = None,
        top_p: float = None,
        do_sample: bool = None,
        max_new_tokens: int = None,
    ):
        """
        Initialize the SparseVLMRewardWrapper.

        Args:
            env: The environment to wrap
            prompt: Question prompt (e.g., "Does the fridge in the scene open? Answer yes or no.")
            camera_name: Name of the camera to use for observations
            vlm_server_url: URL of the VLM reward server endpoint
            update_interval: Minimum time (in seconds) between VLM inference updates
            reward_scale: Scaling factor for the VLM-based reward (default: 1.0)
            temperature: Sampling temperature (optional, uses server default if None)
            top_p: Top-p sampling (optional, uses server default if None)
            do_sample: Whether to use sampling (optional, uses server default if None)
            max_new_tokens: Maximum new tokens (optional, uses server default if None)
        """
        super().__init__(env)
        import threading

        self.prompt = prompt
        self.camera_name = camera_name
        self.vlm_server_url = vlm_server_url
        self.update_interval = update_interval
        self.reward_scale = reward_scale

        # Optional VLM parameters
        self.temperature = temperature
        self.top_p = top_p
        self.do_sample = do_sample
        self.max_new_tokens = max_new_tokens

        # Initialize variables
        self.cached_reward = 0  # Sparse reward: 0 or 1 (or -1 for uncertain)
        self.cached_answer = ""
        self.cached_confidence = "uncertain"
        self.last_image = None
        self.last_update_time = 0.0

        # Threading setup
        self._stop_thread = False
        self._thread_lock = threading.Lock()
        self._update_thread = threading.Thread(target=self._continuous_update, daemon=True)
        self._update_thread.start()

    def _get_vlm_reward(self, image: Image.Image) -> dict:
        """
        Query the VLM server for yes/no reward.

        Args:
            image: PIL Image to evaluate

        Returns:
            dict: {
                "reward": int (1 for yes, 0 for no, -1 for uncertain),
                "answer": str (generated text),
                "confidence": str ("certain" or "uncertain")
            }
        """
        try:
            # Convert image to base64 (encode as JPEG for VLM server)
            import io
            import base64
            buffered = io.BytesIO()
            image.save(buffered, format="JPEG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

            # Prepare the request data
            data = {
                "image": img_base64,
                "prompt": self.prompt,
            }

            # Add optional parameters if specified
            if self.temperature is not None:
                data["temperature"] = self.temperature
            if self.top_p is not None:
                data["top_p"] = self.top_p
            if self.do_sample is not None:
                data["do_sample"] = self.do_sample
            if self.max_new_tokens is not None:
                data["max_new_tokens"] = self.max_new_tokens

            # Send request to VLM server
            response = requests.post(self.vlm_server_url, json=data, timeout=10.0)
            response.raise_for_status()

            # Extract result
            result = response.json()
            return result

        except Exception as e:
            print(f"Error getting VLM reward: {str(e)}")
            # Try to get more detailed error info from response
            try:
                import traceback
                print(f"Traceback: {traceback.format_exc()}")
                if hasattr(e, 'response') and e.response is not None:
                    print(f"Response status code: {e.response.status_code}")
                    print(f"Response text: {e.response.text}")
            except:
                pass
            # Return last known values on error
            return {
                "reward": self.cached_reward,
                "answer": self.cached_answer,
                "confidence": self.cached_confidence,
            }

    def _continuous_update(self):
        """Background thread function to continuously update the VLM reward."""
        while not self._stop_thread:
            current_time = time.time()
            if current_time - self.last_update_time >= self.update_interval:
                if self.last_image is not None:
                    try:
                        image = self.last_image.copy()
                        if len(image.shape) == 4:  # batched image
                            image = image[0]
                        if isinstance(image, np.ndarray):
                            image = Image.fromarray(image.astype('uint8'), 'RGB')

                        result = self._get_vlm_reward(image)

                        with self._thread_lock:
                            self.cached_reward = result["reward"]
                            self.cached_answer = result["answer"]
                            self.cached_confidence = result["confidence"]
                            self.last_update_time = current_time

                    except Exception as e:
                        print(f"Error in continuous VLM update: {str(e)}")
                else:
                    pass

            # Small sleep to prevent excessive CPU usage
            time.sleep(0.05)

    def reward(self, reward: float) -> tuple:
        """
        Compute the reward using cached VLM result.

        Args:
            reward: Original environment reward

        Returns:
            tuple: (total_reward, vlm_reward, vlm_answer, vlm_confidence)
        """
        with self._thread_lock:
            vlm_reward = self.cached_reward
            vlm_answer = self.cached_answer
            vlm_confidence = self.cached_confidence

        # Convert VLM reward: 1 -> 1.0, 0 -> 0.0, -1 (uncertain) -> 0.0
        vlm_reward_value = max(0, vlm_reward)  # Treat uncertain as 0

        total_reward = reward + self.reward_scale * vlm_reward_value
        return total_reward, vlm_reward_value, vlm_answer, vlm_confidence

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.last_image = obs["images"][self.camera_name].copy()

        # Reset cached values
        with self._thread_lock:
            self.cached_reward = 0
            self.cached_answer = ""
            self.cached_confidence = "uncertain"

        return obs, info

    def step(self, action):
        """Modifies the :attr:`env` :meth:`step` reward using :meth:`self.reward`."""
        observation, reward, terminated, truncated, info = self.env.step(action)

        # Update image for background thread
        self.last_image = observation["images"][self.camera_name]

        # Compute reward using cached VLM result
        reward, vlm_reward, vlm_answer, vlm_confidence = self.reward(reward)

        # Add VLM information to info dict
        info["vlm_reward"] = vlm_reward
        info["vlm_answer"] = vlm_answer
        info["vlm_confidence"] = vlm_confidence

        # Set succeed flag based on VLM reward
        if vlm_reward >= 1.0:
            info["succeed"] = True
            terminated = True  # End episode on success
        else:
            info["succeed"] = False

        return observation, reward, terminated, truncated, info

    def close(self):
        """Clean up resources when environment is closed."""
        self._stop_thread = True
        if self._update_thread.is_alive():
            self._update_thread.join(timeout=1.0)
        super().close()


class AffordanceWrapper(gym.Wrapper):
    """
    This wrapper uses affordance trajectory to guide the agent
    Two-stage affordance
    1. Pre-grasp trajectory
        Goal: Reach the pre-grasp pose and grasp
        Reward:
            Task reward : 1 if grasp is successful
            Affordance reward : pre-grasp trajectory is given (Npre, 6)
                                encourage the agent to visit trajectory waypoints
                                gaussian reward based on distance to target waypoint
    2. Post-grasp trajectory
    """
    def __init__(self, env):
        super().__init__(env)
        # Initialize affordance trajectory
        self.grasp_pose = None
        self.pre_traj = None
        self.post_traj = None
        self.pre_visited = None
        self.post_visited = None

        self.dist_threshold = 0.05 # TODO

        # Multi-Stage
        self.scale = 0.05
        self.stage = 0
        self.max_pre_grasp_length = 75
        self.pre_grasp_step = 0
        self.gripper_pose_threshold = 0.1

    def reset(self, **kwargs):
        # Reset the gripper
        requests.post(self.url + "open_gripper")
        time.sleep(1.0)

        # if self.stage == 1: # when the stage is 1, reset the pose
        if self.stage == 1: # when the stage is 1, re
            reset_pose = np.zeros((7,))
            reset_pose[:3] = self.grasp_pose[:3] # x, y, z
            reset_pose[3:] = R.from_euler("xyz", self.grasp_pose[3:]).as_quat() #
            arr = np.array(reset_pose).astype(np.float32)
            data = {"arr": arr.tolist()}
            requests.post(self.url + "pose", json=data)
            time.sleep(2.0)


        # Reset stage
        self.stage = 0
        self.pre_grasp_step = 0


        obs, info = self.env.reset(**kwargs)
        time.sleep(2.0)

        # Get affordance trajectory
        ps = requests.post(self.url + "getaffordance").json()
        self.grasp_pose = np.array(ps["grasp_pose"]) # x, y, z, r, p, y
        self.pre_traj = np.array(ps["pre_traj"]) # (N, 6)
        self.post_traj = np.array(ps["post_traj"]) # (N, 6)
        self.pre_visited = [False] * self.pre_traj.shape[0]
        self.post_visited = [False] * self.post_traj.shape[0]

        # TODO
        self.grasp_pose[:3] = np.array([0.7386425275196229,0.2571240389863525,1.0385435180520572])


        print(f"Grasp pose from Affordance: {self.grasp_pose}")

        # Grasp Reward
        self.grasp_attempt_count = 0
        self.is_grasping = False
        self.grasp_evaluated = False
        # self.last_gripper_pos = obs["state"][0, 0]
        return obs, info

    def step(self, action):
        new_action = action.copy()
        if self.stage == 0: # Do not grasp
            new_action[-1] = -1.0
            new_action[3:-1] = 0.0 # Do not rotate
        elif self.stage == 1: # Grasp
            new_action[-1] = 1.0
            # TODO
            # new_action[0] = -1.0
            # new_action[1] = -0.2
        else:
            raise ValueError("Invalid stage")

        # obs, reward, done, terminated, info = self.env.step(action)
        obs, reward, done, terminated, info = self.env.step(new_action)
        gripper_pose = obs["state"]["gripper_pose"]


        info["stage"] = self.stage
        # Stage 0: Pre-grasp
        if self.stage == 0: # pre-grasp stage
            # Discourage grasping
            # if gripper_pose > self.gripper_pose_threshold: # gripper closed
            #     done = True
            #     terminated = True
            if self.pre_traj is not None and self.pre_visited is not None:
                cur_point = obs["state"]["tcp_pose"][:3] # x, y, z
                cur_quat = obs["state"]["tcp_pose"][3:] # quat
                cur_rpy = quat_2_euler(cur_quat)
                target_waypoint = self.grasp_pose[:3] # x, y, z
                target_rpy = self.grasp_pose[3:] # r, p, y
                dist = np.linalg.norm(cur_point - target_waypoint)
                affordance_dist_reward = np.exp(-0.5 * (dist / self.scale) ** 2) # [0, 1]
                affordance_reward = affordance_dist_reward # TODO


                # rotation reward using cosine similarity
                # cos_sim = np.dot(target_rpy, cur_rpy) / (np.linalg.norm(target_rpy) * np.linalg.norm(cur_rpy))
                # print("cur_point: ", cur_point, "target_waypoint: ", target_waypoint)
                # print("cur_rpy: ", cur_rpy, "target_rpy: ", target_rpy, "dist: ", dist)
                # affordance_rot_reward = (cos_sim + 1) / 2
                # affordance_reward = 0.5 * affordance_dist_reward + 0.5 * affordance_rot_reward

                info["affordance_reward"] = affordance_reward
                info["succeed"] = False
                info["task_reward"] = 0.0 # TODO

                reward = affordance_reward

                # xy_dist = np.linalg.norm(cur_point[:2] - target_waypoint[:2])
                # print("cur_point: ", cur_point)
                # print("target_waypoint: ", target_waypoint)
                # print("xy_dist: ", xy_dist)
                # if dist < self.dist_threshold:
                # if dist < self.dist_threshold:
                if dist < self.dist_threshold and cur_point[0] > 0.735:
                    # close gripper and move to stage 1
                    requests.post(self.url + "close_gripper")
                    time.sleep(5.0)
                    self.stage = 1

                    print("Update After Grasp Param")
                    requests.post(self.url + "update_param", json=self.config.AFTER_GRASP_PARAM)

                if self.pre_grasp_step >= self.max_pre_grasp_length:
                    done = True
                    terminated = True
                    reward = -5.0 # penalty for not reaching the pre-grasp pose
                self.pre_grasp_step += 1

        # Stage 1: Post-grasp
        elif self.stage == 1:
            # Encourage grasping
            gripper_pose = obs["state"]["gripper_pose"] # [0, 1]
            if gripper_pose < self.gripper_pose_threshold: # gripper fully closed
                done = True
                terminated = True
                info["succeed"] = False

            if self.post_traj is not None and self.post_visited is not None:

                # encourage action to be new_action[:3] = [-1.0, -0.5, 0]
                target_action = np.array([-1.0, -0.3, 0.0])
                action_dist = np.linalg.norm(action[:3] - target_action)
                affordance_action_reward = np.exp(-0.5 * (action_dist / self.scale) ** 2)
                affordance_reward = affordance_action_reward


                info["affordance_reward"] = affordance_reward
                info["succeed"] = False


                reward = affordance_reward

                cur_point = obs["state"]["tcp_pose"][:3] # x, y, z
                if cur_point[1] < self.grasp_pose[1] - 0.35:
                    info["succeed"] = True
                    done = True
                    info["task_reward"] = 1.0
                else:
                    info["task_reward"] = 0.0


                # # decide succeed using task reward
                # task_reward = info.get("task_reward", None)
                # if task_reward is not None:
                #     # print(f"Task Reward: {task_reward}")
                #     info["succeed"] = task_reward > 0.95 # TODO
                #     reward += task_reward
                # else:
                #     info["succeed"] = False

                if info["succeed"]:
                    done = True
        else:
            raise ValueError("Invalid stage")

        return obs, reward, done, terminated, info


class AffordanceExecuteWrapper(gym.Wrapper):
    """
    This wrapper uses affordance trajectory to guide the agent
    Two-stage affordance
    1. Pre-grasp trajectory
        Goal: Reach the pre-grasp pose and grasp
        Reward:
            Task reward : 1 if grasp is successful
            Affordance reward : pre-grasp trajectory is given (Npre, 6)
                                encourage the agent to visit trajectory waypoints
                                gaussian reward based on distance to target waypoint
    2. Post-grasp trajectory
    """
    def __init__(self, env):
        super().__init__(env)
        # Initialize affordance trajectory
        self.grasp_pose = None
        self.pre_traj = None
        self.post_traj = None
        self.pre_visited = None
        self.post_visited = None

    def reset(self, **kwargs):
        # Reset the gripper
        requests.post(self.url + "open_gripper")
        time.sleep(1.0)

        # if self.stage == 1: # when the stage is 1, reset the pose




        obs, info = self.env.reset(**kwargs)
        time.sleep(2.0)

        # Get affordance trajectory
        ps = requests.post(self.url + "getaffordance").json()
        self.grasp_pose = np.array(ps["grasp_pose"]) # x, y, z, r, p, y
        self.pre_traj = np.array(ps["pre_traj"]) # (N, 6)
        self.post_traj = np.array(ps["post_traj"]) # (N, 6)

        reset_pose = np.zeros((7,))
        reset_pose[:3] = self.pre_traj[0, :3] # x, y, z
        reset_pose[3:] = R.from_euler("xyz", self.pre_traj[0, 3:]).as_quat() #
        arr = np.array(reset_pose).astype(np.float32)
        data = {"arr": arr.tolist()}
        requests.post(self.url + "pose", json=data)
        time.sleep(2.0)

        # Reset stage
        self.stage = 0
        self.cnt = 0
        print(f"Grasp pose from Affordance: {self.grasp_pose}")
        return obs, info

    def step(self, action):
        new_action = action.copy()

        if self.stage == 0: # pre-grasp stage
            new_action[-1] = -1.0 # do not grasp

        if self.cnt == len(self.pre_traj):
            self.stage = 1
            new_action[-1] = 1.0 # grasp




        # obs, reward, done, terminated, info = self.env.step(action)
        obs, reward, done, terminated, info = self.env.step(new_action)
        gripper_pose = obs["state"]["gripper_pose"]


        info["stage"] = self.stage
        # Stage 0: Pre-grasp
        if self.stage == 0: # pre-grasp stage
            # Discourage grasping
            # if gripper_pose > self.gripper_pose_threshold: # gripper closed
            #     done = True
            #     terminated = True
            if self.pre_traj is not None and self.pre_visited is not None:
                cur_point = obs["state"]["tcp_pose"][:3] # x, y, z
                cur_quat = obs["state"]["tcp_pose"][3:] # quat
                cur_rpy = quat_2_euler(cur_quat)
                target_waypoint = self.grasp_pose[:3] # x, y, z
                target_rpy = self.grasp_pose[3:] # r, p, y
                dist = np.linalg.norm(cur_point - target_waypoint)
                affordance_dist_reward = np.exp(-0.5 * (dist / self.scale) ** 2) # [0, 1]
                affordance_reward = affordance_dist_reward # TODO


                # rotation reward using cosine similarity
                # cos_sim = np.dot(target_rpy, cur_rpy) / (np.linalg.norm(target_rpy) * np.linalg.norm(cur_rpy))
                # print("cur_point: ", cur_point, "target_waypoint: ", target_waypoint)
                # print("cur_rpy: ", cur_rpy, "target_rpy: ", target_rpy, "dist: ", dist)
                # affordance_rot_reward = (cos_sim + 1) / 2
                # affordance_reward = 0.5 * affordance_dist_reward + 0.5 * affordance_rot_reward

                info["affordance_reward"] = affordance_reward
                info["succeed"] = False
                info["task_reward"] = 0.0 # TODO

                reward = affordance_reward

                # xy_dist = np.linalg.norm(cur_point[:2] - target_waypoint[:2])
                # print("cur_point: ", cur_point)
                # print("target_waypoint: ", target_waypoint)
                # print("xy_dist: ", xy_dist)
                # if dist < self.dist_threshold:
                # if dist < self.dist_threshold:
                if dist < self.dist_threshold and cur_point[0] > 0.735:
                    # close gripper and move to stage 1
                    requests.post(self.url + "close_gripper")
                    time.sleep(5.0)
                    self.stage = 1

                    print("Update After Grasp Param")
                    requests.post(self.url + "update_param", json=self.config.AFTER_GRASP_PARAM)

                if self.pre_grasp_step >= self.max_pre_grasp_length:
                    done = True
                    terminated = True
                    reward = -5.0 # penalty for not reaching the pre-grasp pose
                self.pre_grasp_step += 1

        # Stage 1: Post-grasp
        elif self.stage == 1:
            # Encourage grasping
            gripper_pose = obs["state"]["gripper_pose"] # [0, 1]
            if gripper_pose < self.gripper_pose_threshold: # gripper fully closed
                done = True
                terminated = True
                info["succeed"] = False

            if self.post_traj is not None and self.post_visited is not None:

                # encourage action to be new_action[:3] = [-1.0, -0.5, 0]
                target_action = np.array([-1.0, -0.3, 0.0])
                action_dist = np.linalg.norm(action[:3] - target_action)
                affordance_action_reward = np.exp(-0.5 * (action_dist / self.scale) ** 2)
                affordance_reward = affordance_action_reward


                info["affordance_reward"] = affordance_reward
                info["succeed"] = False


                reward = affordance_reward

                cur_point = obs["state"]["tcp_pose"][:3] # x, y, z
                if cur_point[1] < self.grasp_pose[1] - 0.15:
                    info["succeed"] = True
                    done = True
                    info["task_reward"] = 1.0
                else:
                    info["task_reward"] = 0.0


                # # decide succeed using task reward
                # task_reward = info.get("task_reward", None)
                # if task_reward is not None:
                #     # print(f"Task Reward: {task_reward}")
                #     info["succeed"] = task_reward > 0.95 # TODO
                #     reward += task_reward
                # else:
                #     info["succeed"] = False

                if info["succeed"]:
                    done = True
        else:
            raise ValueError("Invalid stage")

        return obs, reward, done, terminated, info
