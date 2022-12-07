from importlib import import_module

import torch

import utils
from arguments import argument_parser
from trainer import Trainer


# This is main funciton using no parallelism, just for style suggestion
def main():
    args = argument_parser()
    cfg = utils.get_cfg(args)
    device = None

    # call module module
    model_module = import_module(f"models.{cfg.model.module}")
    # initialize model object in the model module
    model = getattr(model_module, f"{cfg.model.object}")(cfg.model.arguments)

    optimizer = None
    scheduler = None
    criterion = None

    train_dataset = None
    val_dataset = None
    test_dataset = None

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=1, num_workers=0
    )
    val_loader = torch.utils.data.DataLoader(
        dataset=val_dataset, batch_size=1, num_workers=0
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=1, num_workers=0
    )
    trainer = Trainer(
        cfg=cfg,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
    )

    if cfg.mode == "train":
        trainer.train()
    elif cfg.mode == "test":
        trainer.test()
    else:
        raise ValueError("specify proper trainer mode")

    return



if __name__ == "__main__":
    main()
