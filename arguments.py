import argparse


def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_data_path", type=str)
    parser.add_argument("--cfg_model_path", type=str)
    parser.add_argument("--cfg_optimizer_path", type=str)
    parser.add_argument("--cfg_trainer_path", type=str)

    args = parser.parse_args()
    return args