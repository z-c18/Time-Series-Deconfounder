import os
import argparse
import pickle
import logging

import numpy as np

from simulated_autoregressive_new import AutoregressiveSimulation



def init_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gamma", default=0.6, type=float)
    parser.add_argument("--num_simulated_hidden_confounders", default=1, type=int)
    parser.add_argument("--num_substitute_hidden_confounders", default=1, type=int)
    parser.add_argument("--results_dir", default='results')
    parser.add_argument("--exp_name", default='test_tsd_gamma_0.6')
    parser.add_argument("--b_hyperparm_tuning", default=False)
    return parser.parse_args()


if __name__ == '__main__':
    args = init_arg()
    # Simulate dataset
    np.random.seed(100)
    print(1)
    autoregressive = AutoregressiveSimulation(args.gamma, args.num_simulated_hidden_confounders)
    print(2)
    dataset = autoregressive.generate_dataset(500, 31)
    print(dataset['treatments'][1:5])
    print(3)
