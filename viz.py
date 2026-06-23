"""
evaluate the trained world model and make all the readme pictures.

    python viz.py

what it does:
  1. loss curve (prediction loss vs the variance term)
  2. linear probe: freeze the encoder, fit a linear map latent -> (x,y). if the
     latent really encodes the ball's position this should be near-perfect. it
     also proves the representation didn't collapse.
  3. a 2d pca of the latents coloured by the ball's true x position.
  4. a multi-step latent rollout: encode the first frame, then run ONLY the
     predictor forward using the action sequence (never looking at the real
     frames again), decode each predicted latent with the probe, and compare the
     imagined trajectory to the true one.
"""

import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge

from env import make_transitions, make_episode, render, IMG
from model import JEPAWorldModel


def load_model():
    ckpt = torch.load("out/ckpt.pt", map_location="cpu", weights_only=False)
    model = JEPAWorldModel(dim=ckpt["dim"])
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def plot_loss(out="assets/loss_curve.png"):
    step, pred, var = [], [], []
    with open("out/loss.csv") as f:
        r = csv.reader(f); next(r)
        for row in r:
            step.append(int(row[0])); pred.append(float(row[2])); var.append(float(row[3]))
    plt.figure(figsize=(7, 4))
    plt.plot(step, pred, label="latent prediction loss")
    plt.plot(step, var, label="variance term (anti-collapse)")
    plt.xlabel("step"); plt.ylabel("loss"); plt.title("jepa world model training")
    plt.legend(); plt.grid(alpha=0.25); plt.tight_layout()
    plt.savefig(out, dpi=130); plt.close(); print("wrote", out)


@torch.no_grad()
def fit_probe(model):
    # frozen encoder -> latents, fit a linear map to true (x,y)
    f0, _, _, st = make_transitions(8000, np.random.default_rng(7))
    z = model.encoder(torch.tensor(f0)).numpy()
    xy = st[:, :2]
    n = 6000
    probe = Ridge(alpha=1.0).fit(z[:n], xy[:n])
    pred = probe.predict(z[n:])
    err_px = np.mean(np.linalg.norm((pred - xy[n:]) * (IMG - 1), axis=1))
    r2 = probe.score(z[n:], xy[n:])
    return probe, z, xy, pred, xy[n:], r2, err_px


def plot_probe(pred, true, out="assets/probe.png"):
    plt.figure(figsize=(5, 5))
    plt.scatter(true[:, 0], pred[:, 0], s=4, alpha=0.3, label="x")
    plt.scatter(true[:, 1], pred[:, 1], s=4, alpha=0.3, label="y")
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("true position"); plt.ylabel("decoded from latent")
    plt.title("linear probe: latent -> position"); plt.legend()
    plt.tight_layout(); plt.savefig(out, dpi=130); plt.close(); print("wrote", out)


def plot_pca(z, xy, out="assets/latent_pca.png"):
    p = PCA(n_components=2).fit_transform(z[:3000])
    plt.figure(figsize=(5.5, 4.5))
    sc = plt.scatter(p[:, 0], p[:, 1], c=xy[:3000, 0], cmap="viridis", s=6)
    plt.colorbar(sc, label="true x position")
    plt.title("latent space (pca), coloured by x"); plt.xlabel("pc1"); plt.ylabel("pc2")
    plt.tight_layout(); plt.savefig(out, dpi=130); plt.close(); print("wrote", out)


@torch.no_grad()
def _rollout_one(model, probe, frame0, actions):
    """encode the first pair, then run only the predictor forward."""
    z = model.encoder(torch.tensor(frame0[None]))
    out = []
    for a in actions:
        z = model.predictor(z, torch.tensor(a[None]))
        out.append(probe.predict(z.numpy())[0])
    return np.array(out)


@torch.no_grad()
def plot_rollout(model, probe, k=10, out="assets/rollout.png"):
    frame0, actions, true_states = make_episode(k, np.random.default_rng(3))
    pred_xy = _rollout_one(model, probe, frame0, actions)
    true_xy = true_states[:, :2]

    plt.figure(figsize=(5, 5))
    plt.plot(true_xy[:, 0], true_xy[:, 1], "o-", label="true path", ms=4)
    plt.plot(pred_xy[:, 0], pred_xy[:, 1], "s--", label="imagined (latent rollout)", ms=4)
    plt.scatter([true_xy[0, 0]], [true_xy[0, 1]], c="green", s=90, zorder=5, label="start")
    plt.xlim(0, 1); plt.ylim(0, 1); plt.gca().invert_yaxis()
    plt.title(f"{k}-step rollout, no frames after step 0"); plt.legend()
    plt.tight_layout(); plt.savefig(out, dpi=130); plt.close(); print("wrote", out)


@torch.no_grad()
def plot_rollout_error(model, probe, k=12, episodes=300, out="assets/rollout_error.png"):
    """how far can the model imagine before it drifts? average position error at
    each horizon, over many episodes, vs a 'ball never moves' baseline."""
    jepa_err = np.zeros(k)
    base_err = np.zeros(k)
    for e in range(episodes):
        frame0, actions, true_states = make_episode(k, np.random.default_rng(1000 + e))
        true_xy = true_states[:, :2]
        pred_xy = _rollout_one(model, probe, frame0, actions)
        start = true_xy[0]
        jepa_err += np.linalg.norm(pred_xy - true_xy, axis=1)
        base_err += np.linalg.norm(start[None] - true_xy, axis=1)
    jepa_err = jepa_err / episodes * (IMG - 1)   # to pixels
    base_err = base_err / episodes * (IMG - 1)

    steps = np.arange(1, k + 1)
    plt.figure(figsize=(7, 4))
    plt.plot(steps, jepa_err, "o-", label="jepa latent rollout")
    plt.plot(steps, base_err, "s--", label="baseline (assume no motion)")
    plt.xlabel("rollout horizon (steps imagined)"); plt.ylabel("mean position error (px)")
    plt.title("how far the world model can imagine"); plt.legend(); plt.grid(alpha=0.25)
    plt.tight_layout(); plt.savefig(out, dpi=130); plt.close(); print("wrote", out)
    print(f"rollout error @1 step = {jepa_err[0]:.2f}px, @{k} steps = {jepa_err[-1]:.2f}px "
          f"(baseline @{k}: {base_err[-1]:.2f}px)")


if __name__ == "__main__":
    os.makedirs("assets", exist_ok=True)
    model = load_model()
    plot_loss()
    probe, z, xy, pred, true, r2, err_px = fit_probe(model)
    print(f"linear probe  R^2 = {r2:.3f}   mean position error = {err_px:.2f} px")
    plot_probe(pred, true)
    plot_pca(z, xy)
    plot_rollout(model, probe)
    plot_rollout_error(model, probe)
