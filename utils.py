from omegaconf import OmegaConf
import neptune.new as neptune

def init_neptune(cfg):
    run = neptune.init(project="chanhui-lee/example")
    run["cfg"] = cfg
    return run

def get_cfg(args):
    cfg = OmegaConf.load(args.cfg_path)

    structured_cfg = OmegaConf.structured(cfg)
    return structured_cfg
