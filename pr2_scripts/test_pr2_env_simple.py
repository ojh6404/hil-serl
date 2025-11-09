#!/usr/bin/env python3
"""
Simple test to check if PR2 env sends commands and robot moves.
"""

import os
# Disable display-related issues
os.environ['QT_QPA_PLATFORM'] = 'offscreen'

import sys
import time
import numpy as np

sys.path.insert(0, '/home/leus/prog/hil-serl/serl_robot_infra')
from pr2_envs.envs.pr2_env import PR2Env, DefaultEnvConfig

print("=" * 60)
print("Simple PR2 Environment Test")
print("=" * 60)

# Create simple config
config = DefaultEnvConfig()
config.SERVER_URL = "http://127.0.0.1:5000/"
config.CAMERAS = {}  # No cameras for simple test
config.TARGET_POSE = np.array([0.7, 0.3, 1.1, 1.5, 0, 0.0])
config.RESET_POSE = config.TARGET_POSE + np.array([-0.3, 0, 0, 0, 0, 0])
config.ABS_POSE_LIMIT_LOW = config.TARGET_POSE + np.array([-0.5, -0.4, -0.2, -0.5, -0.5, -0.5])
config.ABS_POSE_LIMIT_HIGH = config.TARGET_POSE + np.array([0.1, 0.1, 0.2, 0.5, 0.5, 0.5])
config.ACTION_SCALE = (0.02, 0.1, 1)  # Small movements
config.DISPLAY_IMAGE = False
config.MAX_EPISODE_LENGTH = 100
config.RANDOM_RESET = False

print("\nCreating environment...")
env = PR2Env(config=config, fake_env=False, save_video=False)

print("Environment created!")
print(f"Action space: {env.action_space}")
print(f"Action space shape: {env.action_space.shape}")

print("\nResetting environment...")
obs = env.reset()
print("Reset complete!")

print("\nCurrent state:")
env._update_currpos()
print(f"  Position (xyz): {env.currpos[:3]}")
print(f"  Orientation (quat): {env.currpos[3:]}")
print(f"  Gripper: {env.curr_gripper_pos}")

print("\n" + "=" * 60)
print("Sending 5 small movements")
print("Watch the robot! It should move slightly.")
print("=" * 60)

input("Press ENTER to start or Ctrl+C to abort...")

for i in range(5):
    print(f"\nStep {i+1}/5:")

    # Small random action
    action = np.random.uniform(-0.3, 0.3, size=env.action_space.shape)
    print(f"  Action: {action}")

    # Get state before
    env._update_currpos()
    pos_before = env.currpos[:3].copy()

    # Execute action
    start_time = time.time()
    obs, reward, done, terminated, info = env.step(action)
    elapsed = time.time() - start_time

    # Get state after
    env._update_currpos()
    pos_after = env.currpos[:3].copy()

    delta = pos_after - pos_before
    print(f"  Position delta: {delta}")
    print(f"  Delta magnitude: {np.linalg.norm(delta):.4f} m")
    print(f"  Step time: {elapsed:.3f} s")

    if np.linalg.norm(delta) < 0.001:
        print("  ⚠ WARNING: Robot barely moved!")
    else:
        print("  ✓ Robot moved")

    time.sleep(0.5)

print("\n" + "=" * 60)
print("Test completed!")
print("=" * 60)

env.close()
