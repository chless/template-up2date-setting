import neptune.new as neptune
from omegaconf import OmegaConf


def init_neptune(cfg):
    run = neptune.init(project="chless/lg-mi-lithopred")
    run["cfg"] = cfg
    run["data"].track_files(cfg.data.path)
    return run

def get_total_cfg(args):
    cfg_data = OmegaConf.load(args.cfg_path_data)
    cfg_optimizer = OmegaConf.load(args.cfg_path_optimizer)
    cfg_model = OmegaConf.load(args.cfg_path_model)
    cfg_trainer = OmegaConf.load(args.cfg_path_trainer)
    cfg = OmegaConf.create(
        {
            "data": cfg_data,
            "model": cfg_model,
            "trainer": cfg_trainer,
            "optimizer": cfg_optimizer
            }
    )
    return cfg
