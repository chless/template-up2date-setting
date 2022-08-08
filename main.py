from arguments import argument_parser
from omegaconf import OmegaConf
import utils


def main():
    args = argument_parser()
    cfg = utils.get_total_cfg(args)
    return args, cfg

def worker():
    return