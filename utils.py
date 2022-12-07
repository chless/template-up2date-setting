from omegaconf import OmegaConf


def get_cfg(args):
    cfg = OmegaConf.load(args.cfg_path_main)
    return cfg
