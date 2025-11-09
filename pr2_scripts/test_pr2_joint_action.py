#!/usr/bin/env python3
"""
Test script for PR2 Joint Action Controller.
Tests the pr2_server_joint_action.py server.
"""

import requests
import json
import time
import numpy as np

# Configuration
SERVER_URL = "http://127.0.0.1:5000/"

def test_connection():
    """Test basic connection to server"""
    print("=" * 60)
    print("Test 1: Server Connection")
    print("=" * 60)
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
    print("Test 2: Get Robot State")
    print("=" * 60)
    try:
        response = requests.post(f"{SERVER_URL}getstate", timeout=5)
        if response.status_code == 200:
            state = response.json()
            print("✓ Robot state received:")
            print(f"  Pose (xyz + quat): {np.array(state['pose'])}")
            print(f"  Joint positions (7): {np.array(state['q'])[:7]}")
            print(f"  Gripper position: {state['gripper_pos']}")
            print(f"  Effort (7): {np.array(state['effort'])[:7]}")
            return state
        else:
            print(f"✗ Failed: {response.status_code}")
            return None
    except Exception as e:
        print(f"✗ Error: {e}")
        return None

def test_gripper():
    """Test gripper control"""
    print("\n" + "=" * 60)
    print("Test 3: Gripper Control")
    print("=" * 60)

    try:
        print("Opening gripper...")
        response = requests.post(f"{SERVER_URL}open_gripper", timeout=5)
        if response.status_code == 200:
            print(f"  ✓ {response.text}")
        else:
            print(f"  ✗ Failed: {response.status_code}")
        time.sleep(2)

        print("Closing gripper...")
        response = requests.post(f"{SERVER_URL}close_gripper", timeout=5)
        if response.status_code == 200:
            print(f"  ✓ {response.text}")
        else:
            print(f"  ✗ Failed: {response.status_code}")
        time.sleep(2)

        print("Opening gripper again...")
        response = requests.post(f"{SERVER_URL}open_gripper", timeout=5)
        if response.status_code == 200:
            print(f"  ✓ {response.text}")
        else:
            print(f"  ✗ Failed: {response.status_code}")

        print("✓ Gripper test completed!")
        return True

    except Exception as e:
        print(f"✗ Gripper test failed: {e}")
        return False

def test_pose_command():
    """Test sending pose command"""
    print("\n" + "=" * 60)
    print("Test 4: Pose Command (Small Movement)")
    print("=" * 60)
    print("WARNING: Robot will move!")

    response = input("Press ENTER to continue or Ctrl+C to abort...")

    try:
        # Get current pose
        state = get_robot_state()
        if state is None:
            print("Cannot get current pose. Aborting.")
            return False

        current_pose = np.array(state['pose'])
        print(f"\nCurrent pose: {current_pose}")

        # Small movement: +2cm in x direction
        target_pose = current_pose.copy()
        target_pose[0] += 0.02  # 2cm in x
        print(f"Target pose: {target_pose}")

        # Send command
        payload = {'arr': target_pose.tolist()}
        response = requests.post(f"{SERVER_URL}pose", json=payload, timeout=10)

        if response.status_code == 200:
            print(f"✓ Command sent: {response.text}")
            time.sleep(3)

            # Check new pose
            new_state = get_robot_state()
            if new_state:
                new_pose = np.array(new_state['pose'])
                delta = new_pose - current_pose
                print(f"\nMovement delta: {delta[:3]} (xyz)")
                print("✓ Pose command test completed!")
            return True
        else:
            print(f"✗ Command failed: {response.status_code}")
            print(f"Response: {response.text}")
            return False

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_joint_command():
    """Test sending joint command"""
    print("\n" + "=" * 60)
    print("Test 5: Joint Command")
    print("=" * 60)
    print("This will move the arm slightly in joint space")

    response = input("Press ENTER to continue or Ctrl+C to abort...")

    try:
        # Get current joint positions
        state = get_robot_state()
        if state is None:
            print("Cannot get current state. Aborting.")
            return False

        current_q = np.array(state['q'])
        print(f"\nCurrent joint positions: {current_q}")

        # Small change in first joint
        target_q = current_q.copy()
        target_q[0] += 0.05  # 0.05 radians (~3 degrees)
        print(f"Target joint positions: {target_q}")

        # Send joint command
        payload = {'arr': target_q.tolist()}
        response = requests.post(f"{SERVER_URL}joint", json=payload, timeout=10)

        if response.status_code == 200:
            print(f"✓ Joint command sent: {response.text}")
            time.sleep(3)

            # Check new positions
            new_state = get_robot_state()
            if new_state:
                new_q = np.array(new_state['q'])
                delta_q = new_q - current_q
                print(f"\nJoint delta: {delta_q}")
                print("✓ Joint command test completed!")
            return True
        else:
            print(f"✗ Command failed: {response.status_code}")
            return False

    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def test_ros_topics():
    """Check if ROS topics are being published"""
    print("\n" + "=" * 60)
    print("Test 6: ROS Topic Check")
    print("=" * 60)
    print("Run this in a separate terminal:")
    print("  rostopic echo /joint_action_controller/joint_controller_state -n 1")
    print("  rostopic hz /joint_action_controller/joint_controller_state")
    print("\nYou should see:")
    print("  - Pose, q, effort fields in the message")
    print("  - Publishing rate around 30 Hz")

def main():
    print("\n" + "=" * 60)
    print("PR2 Joint Action Controller Test Script")
    print("=" * 60)
    print("\nThis will test the pr2_server_joint_action.py server")
    print("\nMake sure:")
    print("  1. rosrun pr2_controller skrobot_joint_action_controller.py is running")
    print("  2. launch_pr2_server_joint_action.sh is running")
    print("  3. Robot workspace is clear")

    input("\nPress ENTER to start tests or Ctrl+C to abort...\n")

    # Test 1: Connection
    if not test_connection():
        print("\n⚠ Server connection failed. Exiting.")
        return

    # Test 2: Get state
    state = get_robot_state()
    if state is None:
        print("\n⚠ Cannot get robot state. Exiting.")
        return

    # Test 3: Gripper
    response = input("\nTest gripper? (y/n): ")
    if response.lower() == 'y':
        test_gripper()

    # Test 4: Pose command
    response = input("\nTest pose command (robot will move)? (y/n): ")
    if response.lower() == 'y':
        test_pose_command()

    # Test 5: Joint command
    response = input("\nTest joint command (robot will move)? (y/n): ")
    if response.lower() == 'y':
        test_joint_command()

    # Test 6: ROS topics
    print("\n")
    test_ros_topics()

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
