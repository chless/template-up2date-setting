import neptune.new as neptune
from omegaconf import OmegaConf


def init_neptune(cfg):
    run = neptune.init(project="chanhui-lee/example")
    run["cfg"] = cfg
    return run


def get_total_cfg(args):
    cfg_main = OmegaConf.load(args.cfg_path_main)
    cfg_data = OmegaConf.load(cfg_main.cfg_path_data)
    cfg_optimizer = OmegaConf.load(cfg_main.cfg_path_optimizer)
    cfg_model = OmegaConf.load(cfg_main.cfg_path_model)
    cfg_trainer = OmegaConf.load(cfg_main.cfg_path_trainer)
    cfg = OmegaConf.create(
        {
            "main": cfg_main,
            "data": cfg_data,
            "model": cfg_model,
            "trainer": cfg_trainer,
            "optimizer": cfg_optimizer,
        }
    )
    return cfg
