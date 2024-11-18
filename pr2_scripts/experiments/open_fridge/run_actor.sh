export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.1 && \
python ../../train_rlpd.py "$@" \
    --exp_name=open_fridge \
    --checkpoint_path=../../experiments/open_fridge/debug \
    --actor
