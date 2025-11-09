export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.3 && \
python ../../train_rlpd.py "$@" \
    --exp_name=open_fridge \
    --checkpoint_path=first_run \
    --demo_path=/home/leus/ros/affordance_ws/src/hil-serl/pr2_scripts/experiments/open_fridge/demo_data.pkl \
    --learner 
