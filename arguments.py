import argparse


def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_data", type=str)
    parser.add_argument("--cfg_model", type=str)
    parser.add_argument("--cfg_optimizer", type=str)
    parser.add_argument("--cfg_trainer", type=str)

    args = parser.parse_args()
    return args