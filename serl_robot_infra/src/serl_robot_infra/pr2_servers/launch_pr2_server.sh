#!/usr/bin/env bash
source ~/ros/exp_ws/devel/setup.bash
rossetip; rossetmaster pr1040

# Run the second instance of franka_server.py in the background
python pr2_server.py \
    --robot_ip pr1040 \
    --flask_url "127.0.0.1"
