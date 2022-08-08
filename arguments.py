import argparse


def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_path_data", type=str)
    parser.add_argument("--cfg_path_model", type=str)
    parser.add_argument("--cfg_path_optimizer", type=str)
    parser.add_argument("--cfg_path_trainer", type=str)

    args = parser.parse_args()
    return args