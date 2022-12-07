import os

import numpy as np
import torch
from tqdm import tqdm


class Trainer:
    def __init__(
        self,
        cfg,
        model,
        optimizer,
        scheduler,
        criterion,
        train_loader,
        val_loader,
        test_loader,
        device,
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.mode = cfg.mode
        self.num_epochs = cfg.trainer.num_epochs
        self.val_interval = cfg.trainer.val_interval
        self.ckpt_save_path = cfg.trainer.ckpt_save_path
        self.ckpt_load_path = cfg.trainer.ckpt_load_path
        self.device = device

        self.load()


    def train(self, *args, **kwargs) -> float:
        self.model.train()

        if self.ckpt_load_path is not None:
            print(f"loading ckpt from {self.ckpt_load_path}")
            self.load()

        self.model.to(device=self.device)

        epoch_iter = tqdm(range(self.num_epochs))
        train_loss, val_loss = 10000.0, 10000.0
        for e in epoch_iter:
            self.model.train()
            train_loss = self.loop("train", *args, **kwargs)
            epoch_iter.set_description(
                f"lr: {self.scheduler.get_last_lr()[-1]} \
                    | train avg loss={train_loss:.5f} \
                        | val avg loss={val_loss:.5f}"
            )
            if e % self.val_interval == 0 and e > 0:
                self.model.eval()
                self.model.to(device=self.device)
                val_loss = self.loop("val", *args, **kwargs)
                self.save(e)
        return train_loss

    def loop(self, mode, *args, **kwgs):
        loss_sum = 0.0
        data_loader = getattr(self, f"{mode}_loader")
        idx, batch = None, None
        for idx, batch in enumerate(data_loader):
            x, y = batch
            x = x.to(self.device)
            y = y.to(self.device)

            if mode == "train":
                self.optimizer.zero_grad()

            batch_out = self.get_output(self.model, self.criterion, x, y, *args, **kwgs)
            loss = batch_out["loss"].mean()

            if mode == "train":
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

            loss_sum += loss.item()
        return loss_sum / (idx + 1)

    def test(self, *args, **kwargs) -> float:
        self.model.eval()
        self.load()

        self.model.to(device=self.device)
        test_loss = self.loop("test", *args, **kwargs)
        return test_loss

    def save(self, tag=None):
        if not os.path.isdir(self.ckpt_save_path):
            os.makedirs(self.ckpt_save_path)

        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
            },
            os.path.join(self.ckpt_save_path, f"{tag}.pt"),
        )
        return

    def load(self):
        if self.ckpt_load_path is None:
            print("ckpt_load_path is None")
            return
        else:
            print(f"loading from {self.ckpt_load_path}")
            checkpoint = torch.load(self.ckpt_load_path)
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

    def get_output(self, model, criterion, x, y, *args, **kwargs):
        out = dict()
        out["loss"] = criterion(model(x), y)
        return out