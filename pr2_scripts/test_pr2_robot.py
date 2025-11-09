#!/usr/bin/env python3
"""
Test script for PR2 robot basic operations.
This script tests:
1. Connection to PR2 server
2. Getting current robot state
3. Moving the robot with simple actions
4. Gripper control
"""

import sys
import time
import numpy as np

sys.path.insert(0, '/home/leus/prog/hil-serl/pr2_scripts/experiments')
from open_fridge.config import TrainConfig, EnvConfig

def test_server_connection(config):
    """Test basic HTTP connection to PR2 server"""
    import requests
    print("=" * 60)
    print("Testing Server Connection")
    print("=" * 60)

    server_url = config.SERVER_URL
    print(f"Server URL: {server_url}")

    try:
        # Test getstate endpoint
        response = requests.post(f"{server_url}getstate", timeout=5)
        if response.status_code == 200:
            print("✓ Server connection successful!")
            state = response.json()
            print(f"  Current TCP pose: {state.get('pose', 'N/A')}")
            return True
        else:
            print(f"✗ Server returned status code: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Failed to connect to server: {e}")
        return False

def test_environment_creation():
    """Test creating PR2 environment"""
    print("\n" + "=" * 60)
    print("Testing Environment Creation")
    print("=" * 60)

    try:
        config = TrainConfig()
        env = config.get_environment(fake_env=False, classifier=False)
        print("✓ Environment created successfully!")
        print(f"  Observation space: {env.observation_space}")
        print(f"  Action space: {env.action_space}")
        return env
    except Exception as e:
        print(f"✗ Failed to create environment: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_reset(env):
    """Test environment reset"""
    print("\n" + "=" * 60)
    print("Testing Environment Reset")
    print("=" * 60)

    try:
        obs, info = env.reset()
        print("✓ Reset successful!")
        print(f"  Observation keys: {obs.keys()}")
        if 'image' in obs:
            for key, img in obs['image'].items():
                print(f"  Image '{key}' shape: {img.shape}")
        if 'state' in obs:
            print(f"  State shape: {obs['state'].shape}")
        return True
    except Exception as e:
        print(f"✗ Reset failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_small_movements(env, num_steps=5):
    """Test small robot movements"""
    print("\n" + "=" * 60)
    print(f"Testing Small Movements ({num_steps} steps)")
    print("=" * 60)
    print("The robot will make small random movements.")
    print("Make sure the workspace is clear!")

    input("Press ENTER to continue or Ctrl+C to abort...")

    try:
        for i in range(num_steps):
            # Small random action (scaled down for safety)
            action = env.action_space.sample() * 0.1  # Scale down to 10%

            print(f"\nStep {i+1}/{num_steps}")
            print(f"  Action: {action}")

            obs, reward, terminated, truncated, info = env.step(action)

            print(f"  Reward: {reward}")
            print(f"  Terminated: {terminated}, Truncated: {truncated}")

            time.sleep(0.5)  # Small delay between actions

        print("\n✓ Movement test completed successfully!")
        return True
    except Exception as e:
        print(f"\n✗ Movement test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_gripper(env):
    """Test gripper control"""
    print("\n" + "=" * 60)
    print("Testing Gripper Control")
    print("=" * 60)

    input("Press ENTER to test gripper or Ctrl+C to skip...")

    try:
        # Get the base environment to access gripper methods
        base_env = env
        while hasattr(base_env, 'env'):
            base_env = base_env.env

        if hasattr(base_env, 'open_gripper') and hasattr(base_env, 'close_gripper'):
            print("Opening gripper...")
            base_env.open_gripper()
            time.sleep(2)

            print("Closing gripper...")
            base_env.close_gripper()
            time.sleep(2)

            print("Opening gripper again...")
            base_env.open_gripper()

            print("✓ Gripper test completed!")
            return True
        else:
            print("⚠ Gripper methods not found in environment")
            return False
    except Exception as e:
        print(f"✗ Gripper test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_observation_collection(env, num_obs=3):
    """Test collecting observations"""
    print("\n" + "=" * 60)
    print(f"Testing Observation Collection ({num_obs} observations)")
    print("=" * 60)

    try:
        for i in range(num_obs):
            # Zero action (no movement)
            action = np.zeros(env.action_space.shape)
            obs, reward, terminated, truncated, info = env.step(action)

            print(f"\nObservation {i+1}/{num_obs}")
            if 'state' in obs:
                tcp_pose = obs['state'][0, :6]  # Assuming first 6 are TCP pose
                print(f"  TCP pose (xyz, rpy): {tcp_pose}")

            time.sleep(0.5)

        print("\n✓ Observation collection test completed!")
        return True
    except Exception as e:
        print(f"\n✗ Observation collection failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("\n" + "=" * 60)
    print("PR2 Robot Test Script")
    print("=" * 60)
    print("\nThis script will test basic PR2 robot operations.")
    print("Make sure:")
    print("  1. PR2 server is running (launch_pr2_server.sh)")
    print("  2. Robot workspace is clear")
    print("  3. You can emergency stop if needed")

    input("\nPress ENTER to start tests or Ctrl+C to abort...\n")

    # Test 1: Server connection
    config = EnvConfig()
    if not test_server_connection(config):
        print("\n⚠ Server connection failed. Please check:")
        print("  1. PR2 server is running")
        print("  2. Server URL is correct in config.py")
        return

    # Test 2: Environment creation
    env = test_environment_creation()
    if env is None:
        return

    try:
        # Test 3: Reset
        if not test_reset(env):
            return

        # Test 4: Observation collection
        test_observation_collection(env, num_obs=3)

        # Test 5: Small movements
        test_small_movements(env, num_steps=5)

        # Test 6: Gripper
        test_gripper(env)

        print("\n" + "=" * 60)
        print("All Tests Completed!")
        print("=" * 60)

    finally:
        print("\nClosing environment...")
        env.close()
        print("Done!")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTest aborted by user.")
    except Exception as e:
        print(f"\n\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
