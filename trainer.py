import os

import numpy as np
import torch
from tqdm import tqdm


class Trainer:
    def __init__(
        self,
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
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.val_loader = val_loader
        self.num_epochs = cfg.trainer.num_epochs
        self.val_interval = cfg.trainer.val_interval
        self.ckpt_save_path = cfg.trainer.ckpt_save_path
        self.ckpt_load_path = cfg.trainer.ckpt_load_path
        self.metric = cfg.optimizer.criterion
        self.device = device
        self.run = run

        if self.ckpt_load_path != "None":
            print(f"load ckpt from {self.ckpt_load_path}\n")
            self.load(self.ckpt_load_path)
            print(f"loaded ckpt from {self.ckpt_load_path}\n")
        return

    def train(self, get_output, *args, **kwargs) -> float:
        self.model.train()
        epoch_iter = tqdm(range(self.train_epochs))
        train_loss, val_loss = 10000.0, 10000.0
        for e in epoch_iter:
            train_loss = self.loop(get_output, "train", *args, **kwargs)
            self.run["train/loss"].log(train_loss)
            epoch_iter.set_description(
                f"{self.metric} | lr: {self.scheduler.get_last_lr()[-1]} \
                    | train avg loss={train_loss:.5f} \
                        | val avg loss={val_loss:.5f}"
            )
            self.run["train/lr"].log(self.scheduler.get_last_lr())
            if e % self.val_interval == 0 and e > 0:
                val_loss = self.validate(get_output, *args, **kwargs)
                self.run["val/loss"].log(val_loss)
                self.save(e)
        return train_loss

    def loop(self, get_output, mode, *args, **kwgs):
        loss_sum = 0.0
        data_loader = getattr(self, f"{mode}_loader")
        idx, batch = None, None
        for idx, batch in enumerate(data_loader):
            x, y = batch
            x = x.to(self.device)
            y = y.to(self.device)

            if mode == "train":
                self.optimizer.zero_grad()

            batch_out = get_output(self.model, self.criterion, x, y, *args, **kwgs)
            loss = batch_out["loss"].mean()

            if mode == "train":
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

            loss_sum += loss.item()
        return loss_sum / (idx + 1)

    def validate(self, get_output, *args, **kwargs) -> float:
        self.model.eval()
        val_loss = self.loop(get_output, "val", *args, **kwargs)
        return val_loss

    def test(self, get_output, *args, **kwargs) -> float:
        self.model.eval()
        test_loss = self.loop(get_output, "test", *args, **kwargs)
        return test_loss

    def save(self, epoch):
        if not os.path.isdir(self.ckpt_save_path):
            os.makedirs(self.ckpt_save_path)

        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
            },
            os.path.join(self.ckpt_save_path, f"ckpt_{epoch}.pt"),
        )
        return

    def load(self, load_path):
        print(f"loading from {load_path}")
        checkpoint = torch.load(load_path)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        return

    def log_grads(self):
        if not hasattr(self, "grads_label"):
            self.grads_label = []
            for n, p in self.model.named_parameters():
                if p.requires_grad and "bias" not in n and p.grad is not None:
                    self.grads_label.append(n)

        grads_abs = []
        for n, p in self.model.named_parameters():
            if p.requires_grad and "bias" not in n and p.grad is not None:
                grads_abs.append(p.grad.abs().mean().cpu().item())
        grads_abs = np.expand_dims(np.array(grads_abs), axis=0)

        if not hasattr(self, "grads_abs"):
            self.grads_abs = np.empty(
                shape=(0,) + grads_abs.shape[1:], dtype=grads_abs.dtype
            )
        self.grads_abs = np.concatenate((self.grads_abs, grads_abs), axis=0)
        return
