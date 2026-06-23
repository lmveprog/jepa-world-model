"""
the jepa world model.

three pieces:
  - an encoder    : image -> a latent vector
  - a predictor   : (latent_t, action) -> predicted latent_{t+1}
  - a target encoder : an ema (slowly-moving) copy of the encoder

the key idea of jepa: we do NOT reconstruct the next frame in pixels. we encode
the real next frame with the target encoder and ask the predictor to hit that
vector in latent space. predicting pixels wastes capacity on stuff that doesn't
matter (exact blob texture); predicting in latent space lets the model focus on
what actually changes - position and motion.

the obvious failure mode is collapse: if the encoder maps everything to the same
vector, the prediction loss is trivially zero. two things stop that here:
  1. the target encoder is an ema with a stop-gradient (the byol/i-jepa trick),
  2. a small variance regulariser (vicreg style) that keeps each latent
     dimension from flatlining across the batch.
"""

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, dim=64, in_ch=2):
        super().__init__()
        # in_ch = 2: two stacked frames (prev, current) so velocity is visible
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, stride=2, padding=1), nn.GroupNorm(8, 32), nn.SiLU(),  # 28->14
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.GroupNorm(8, 64), nn.SiLU(), # 14->7
            nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.GroupNorm(8, 64), nn.SiLU(), # 7->4
            # NOTE: deliberately no global average pooling here - averaging over
            # space would throw away *where* the ball is, which is the one thing
            # we need. flatten the 4x4 map instead so position survives.
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, dim),
        )

    def forward(self, x):
        return self.net(x)


class Predictor(nn.Module):
    """takes the current latent + the action, predicts the next latent."""

    def __init__(self, dim=64, act_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + act_dim, 256), nn.SiLU(),
            nn.Linear(256, 256), nn.SiLU(),
            nn.Linear(256, dim),
        )

    def forward(self, z, a):
        return self.net(torch.cat([z, a], dim=-1))


def variance_reg(z, eps=1e-4):
    # push every feature's std (across the batch) up towards 1 so the
    # representation can't collapse to a constant
    std = torch.sqrt(z.var(dim=0) + eps)
    return torch.mean(F.relu(1.0 - std))


class JEPAWorldModel(nn.Module):
    def __init__(self, dim=64, act_dim=2, ema=0.99, var_coef=1.0):
        super().__init__()
        self.encoder = Encoder(dim)
        self.predictor = Predictor(dim, act_dim)
        # target encoder starts as a frozen copy of the online encoder
        self.target = copy.deepcopy(self.encoder)
        for p in self.target.parameters():
            p.requires_grad_(False)
        self.ema = ema
        self.var_coef = var_coef

    @torch.no_grad()
    def update_target(self):
        # ema: target = ema*target + (1-ema)*online
        for t, o in zip(self.target.parameters(), self.encoder.parameters()):
            t.lerp_(o, 1.0 - self.ema)

    def forward(self, img0, action, img1):
        z0 = self.encoder(img0)                 # online latent of current frame
        with torch.no_grad():
            z1_target = self.target(img1)       # target latent of next frame (stop-grad)
        z1_pred = self.predictor(z0, action)    # what we think the next latent is

        # prediction loss lives entirely in latent space
        pred_loss = F.smooth_l1_loss(z1_pred, z1_target)
        var_loss = variance_reg(z0) + variance_reg(z1_pred)
        loss = pred_loss + self.var_coef * var_loss
        return loss, pred_loss.detach(), var_loss.detach()
