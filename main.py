from importlib import import_module

import utils
from arguments import argument_parser


def main():
    args = argument_parser()
    cfg = utils.get_total_cfg(args)
    module_model = import_module(f"models.{cfg.model.script}")
    model = getattr(module_model, f"{cfg.model.name}")(cfg.model.architecture)
    return model


def worker():
    return


if __name__ == "__main__":
    main()
