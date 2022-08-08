import neptune.new as neptune


def init_neptune(cfg):
    run = neptune.init(project="chless/lg-mi-lithopred")
    run["cfg"] = cfg
    run["data"].track_files(cfg.data.path)
    return run
