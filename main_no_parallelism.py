from importlib import import_module

import torch

import utils
from arguments import argument_parser
from trainer import Trainer


# This is main funciton using no parallelism, just for style suggestion
def main():
    args = argument_parser()
    cfg = utils.get_total_cfg(args)
    run = utils.init_neptune(cfg)
    device = torch.device(
        f"cuda:{cfg.main.gpu}" if torch.cuda.is_available() else "cpu"
    )

    module_model = import_module(f"models.{cfg.model.script}")
    model = getattr(module_model, f"{cfg.model.name}")(cfg.model.architecture)
    optimizer = None
    scheduler = None
    criterion = None

    train_set = None
    val_set = None
    test_set = None
    train_loader = torch.utils.data.DataLoader(
        dataset=train_set, batch_size=1, num_workers=0
    )
    val_loader = torch.utils.data.DataLoader(
        dataset=val_set, batch_size=1, num_workers=0
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_set, batch_size=1, num_workers=0
    )
    trainer = Trainer(
        model,
        optimizer,
        scheduler,
        criterion,
        train_loader,
        val_loader,
        test_loader,
        cfg,
        device,
        run,
    )

    trainer.train(get_output)
    trainer.test(get_output)

    return


def get_output():
    return


if __name__ == "__main__":
    main()
