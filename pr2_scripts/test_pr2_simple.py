#!/usr/bin/env python3
"""
Simple low-level test script for PR2 server.
Tests direct HTTP communication with the PR2 Flask server.
"""

import requests
import json
import time
import numpy as np

# Configuration
SERVER_URL = "http://127.0.0.1:5000/"  # Change this to your server URL

def test_connection():
    """Test basic connection to server"""
    print("Testing connection to server:", SERVER_URL)
    try:
        response = requests.post(f"{SERVER_URL}getstate", timeout=5)
        if response.status_code == 200:
            print("✓ Connection successful!")
            return True
        else:
            print(f"✗ Server returned error: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"✗ Connection failed: {e}")
        return False

def get_robot_state():
    """Get current robot state"""
    print("\n" + "=" * 60)
    print("Getting Robot State")
    print("=" * 60)
    try:
        response = requests.post(f"{SERVER_URL}getstate", timeout=5)
        if response.status_code == 200:
            state = response.json()
            print("Current robot state:")
            for key, value in state.items():
                if isinstance(value, (list, np.ndarray)):
                    print(f"  {key}: {np.array(value)}")
                else:
                    print(f"  {key}: {value}")
            return state
        else:
            print(f"Failed to get state: {response.status_code}")
            return None
    except Exception as e:
        print(f"Error: {e}")
        return None

def get_current_pose():
    """Get current end-effector pose"""
    print("\n" + "=" * 60)
    print("Getting Current End-Effector Pose")
    print("=" * 60)
    try:
        response = requests.post(f"{SERVER_URL}getpos", timeout=5)
        if response.status_code == 200:
            pose = response.json()
            print(f"Current pose: {pose}")
            return pose
        else:
            print(f"Failed: {response.status_code}")
            return None
    except Exception as e:
        print(f"Error: {e}")
        return None

def test_gripper():
    """Test gripper open/close"""
    print("\n" + "=" * 60)
    print("Testing Gripper")
    print("=" * 60)

    input("Press ENTER to open gripper...")
    try:
        response = requests.post(f"{SERVER_URL}open_gripper", timeout=5)
        print(f"Open gripper: {response.text if response.status_code == 200 else 'Failed'}")
        time.sleep(2)

        input("Press ENTER to close gripper...")
        response = requests.post(f"{SERVER_URL}close_gripper", timeout=5)
        print(f"Close gripper: {response.text if response.status_code == 200 else 'Failed'}")
        time.sleep(2)

        input("Press ENTER to open gripper again...")
        response = requests.post(f"{SERVER_URL}open_gripper", timeout=5)
        print(f"Open gripper: {response.text if response.status_code == 200 else 'Failed'}")

    except Exception as e:
        print(f"Gripper test failed: {e}")

def test_small_movement():
    """Test small movement"""
    print("\n" + "=" * 60)
    print("Testing Small Movement")
    print("=" * 60)
    print("WARNING: Robot will move!")
    print("Make sure workspace is clear!")

    input("Press ENTER to continue or Ctrl+C to abort...")

    # Get current pose
    current_pose = get_current_pose()
    if current_pose is None:
        print("Cannot get current pose. Aborting.")
        return

    # Calculate small movement (move 2cm in x direction)
    if isinstance(current_pose, dict) and 'pose' in current_pose:
        pose_array = np.array(current_pose['pose'])
    else:
        pose_array = np.array(current_pose)

    print(f"Current pose: {pose_array}")

    # Small movement: +2cm in x
    target_pose = pose_array.copy()
    target_pose[0] += 0.02  # 2cm in x direction

    print(f"Target pose: {target_pose}")

    try:
        # Send pose command (PR2 server expects 'arr' key)
        payload = {'arr': target_pose.tolist()}
        response = requests.post(
            f"{SERVER_URL}pose",
            json=payload,
            timeout=10
        )
        if response.status_code == 200:
            print("✓ Movement command sent successfully!")
            time.sleep(2)

            # Check new pose
            new_pose = get_current_pose()
            print(f"New pose: {new_pose}")
        else:
            print(f"✗ Movement failed: {response.status_code}")
            print(f"Response: {response.text}")

    except Exception as e:
        print(f"Error during movement: {e}")

def main():
    print("\n" + "=" * 60)
    print("PR2 Simple Test Script")
    print("=" * 60)
    print(f"\nServer URL: {SERVER_URL}")
    print("\nMake sure PR2 server is running!")

    input("\nPress ENTER to start or Ctrl+C to abort...\n")

    # Test connection
    if not test_connection():
        print("\nCannot connect to server. Please check:")
        print("  1. Server is running (launch_pr2_server.sh)")
        print("  2. Server URL is correct")
        print("  3. Network connectivity")
        return

    # Get robot state
    state = get_robot_state()

    # Get current pose
    pose = get_current_pose()

    # Test gripper
    response = input("\nTest gripper? (y/n): ")
    if response.lower() == 'y':
        test_gripper()

    # Test movement
    response = input("\nTest small movement? (y/n): ")
    if response.lower() == 'y':
        test_small_movement()

    print("\n" + "=" * 60)
    print("Tests Completed!")
    print("=" * 60)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nAborted by user.")
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
