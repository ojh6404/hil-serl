#!/usr/bin/env python3

import os
from tqdm import tqdm
import numpy as np
import copy
import pickle as pkl
import datetime
from absl import app, flags
import time

from experiments.mappings import CONFIG_MAPPING

FLAGS = flags.FLAGS
flags.DEFINE_string("exp_name", None, "Name of experiment corresponding to folder.")

def main(_):
    assert FLAGS.exp_name in CONFIG_MAPPING, 'Experiment folder not found.'
    config = CONFIG_MAPPING[FLAGS.exp_name]()
    env = config.get_environment(fake_env=False, save_video=False, classifier=False)

    obs, info = env.reset()
    print("Reset done")
    transitions = []
    trajectory = []
    returns = 0

    while True:
        actions = np.zeros(env.action_space.sample().shape)
        obs, rew, done, truncated, info = env.step(actions)
        returns += rew
        print("reward:", rew, "done:", done)
        if "intervene_action" in info:
            actions = info["intervene_action"]
            print("Intervene action:", actions)

        # obs = next_obs
        if done:
            if info["succeed"]:
                for transition in trajectory:
                    transitions.append(copy.deepcopy(transition))
            trajectory = []
            returns = 0
            obs, info = env.reset()


if __name__ == "__main__":
    app.run(main)
