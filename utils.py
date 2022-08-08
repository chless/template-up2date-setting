import neptune.new as neptune
from omegaconf import DictConfig


def init_neptune(cfg: DictConfig):
    run = neptune.init(project="chless/lg-mi-lithopred")
    run["cfg"] = cfg
    run["data"].track_files(cfg.data.path)
    return run
