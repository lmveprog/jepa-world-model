"""
train the jepa world model on the bouncing-ball env.

    python train.py --steps 4000

generates a pile of (frame, action, next frame) transitions, then trains the
encoder + predictor with the latent prediction loss (+ ema target update). saves
the checkpoint and a loss log.
"""

import argparse
import csv
import os

import numpy as np
import torch

from env import make_transitions
from model import JEPAWorldModel


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--n_data", type=int, default=40000)
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--out", default="out")
    args = p.parse_args()

    device = pick_device()
    os.makedirs(args.out, exist_ok=True)
    print("device:", device)

    print("generating env data...")
    f0, acts, f1, st1 = make_transitions(args.n_data, np.random.default_rng(0))
    f0 = torch.tensor(f0, device=device)
    f1 = torch.tensor(f1, device=device)
    acts = torch.tensor(acts, device=device)
    N = f0.shape[0]
    print(f"{N} transitions, image {tuple(f0.shape[1:])}")

    model = JEPAWorldModel(dim=args.dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"params: {n_params/1e6:.2f}M")

    log = [("step", "loss", "pred_loss", "var_loss")]
    for step in range(args.steps + 1):
        idx = torch.randint(0, N, (args.batch_size,), device=device)
        loss, pred_l, var_l = model(f0[idx], acts[idx], f1[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
        model.update_target()       # nudge the ema target encoder

        if step % 200 == 0:
            print(f"step {step:5d} | loss {loss.item():.4f} "
                  f"| pred {pred_l.item():.4f} | var {var_l.item():.4f}")
            log.append((step, round(loss.item(), 4),
                        round(pred_l.item(), 4), round(var_l.item(), 4)))

    torch.save({"model": model.state_dict(), "dim": args.dim},
               f"{args.out}/ckpt.pt")
    with open(f"{args.out}/loss.csv", "w", newline="") as fp:
        csv.writer(fp).writerows(log)
    print("saved ->", f"{args.out}/ckpt.pt")


if __name__ == "__main__":
    main()
