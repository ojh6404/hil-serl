export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.3 && \
python ../../train_sac.py "$@" \
    --exp_name=open_fridge \
    --checkpoint_path=first_run \
    --learner 
