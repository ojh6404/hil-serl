#!/usr/bin/env bash
source ~/ros/exp_ws/devel/setup.bash
rossetip; rossetmaster pr1040

# Run the PR2 server with joint action controller
python pr2_server_joint_action.py \
    --robot_ip pr1040 \
    --flask_url "127.0.0.1"
