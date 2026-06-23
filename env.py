"""
a tiny toy "world": a ball bouncing around a 2d box.

the state is (x, y, vx, vy). every step an action nudges the velocity, the ball
moves, and it bounces off the walls. we render it to a small grayscale image (a
soft blob where the ball is). that image is all the model ever sees - it never
gets the true x,y. the point is to learn the dynamics of this world from pixels,
but predict it in latent space (that's the jepa part).
"""

import numpy as np

IMG = 28          # image is IMG x IMG
DT = 0.08         # timestep
BLOB = 1.4        # blob radius in pixels (gaussian sigma)


class BouncingBall:
    def __init__(self, rng=None):
        self.rng = rng or np.random.default_rng()
        self.reset()

    def reset(self):
        self.x, self.y = self.rng.uniform(0.2, 0.8, size=2)
        self.vx, self.vy = self.rng.uniform(-1, 1, size=2)
        return self.state()

    def state(self):
        return np.array([self.x, self.y, self.vx, self.vy], dtype=np.float32)

    def step(self, action):
        # action is a small 2d push on the velocity
        ax, ay = action
        self.vx += 0.6 * ax
        self.vy += 0.6 * ay
        # keep speeds sane
        self.vx = np.clip(self.vx, -2, 2)
        self.vy = np.clip(self.vy, -2, 2)

        self.x += self.vx * DT
        self.y += self.vy * DT

        # bounce off the walls (reflect the velocity, keep pos in bounds)
        for pos, vel in (("x", "vx"), ("y", "vy")):
            p, v = getattr(self, pos), getattr(self, vel)
            if p < 0.05:
                p, v = 0.05, abs(v)
            elif p > 0.95:
                p, v = 0.95, -abs(v)
            setattr(self, pos, p)
            setattr(self, vel, v)
        return self.state()


def render(state):
    """draw the ball as a soft gaussian blob -> (1, IMG, IMG) float image in [0,1]."""
    x, y = state[0], state[1]
    cx, cy = x * (IMG - 1), y * (IMG - 1)
    yy, xx = np.mgrid[0:IMG, 0:IMG]
    img = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * BLOB ** 2))
    return img.astype(np.float32)[None]   # add channel dim


def random_action(rng):
    return rng.uniform(-1, 1, size=2).astype(np.float32)


# we feed the encoder TWO stacked frames, not one. a single frame shows where the
# ball is but not which way it's moving - velocity is invisible in a still image.
# stacking (prev, current) makes velocity observable, which is what lets the
# predictor actually roll the world forward. (same trick as frame-stacking in
# atari rl.)
STACK = 2


def _rollout_frames(env, length, rng):
    """play one episode, return the list of rendered frames and actions taken."""
    s = env.reset()
    frames = [render(s)[0]]
    actions, states = [], []
    for _ in range(length):
        a = random_action(rng)
        s = env.step(a)
        frames.append(render(s)[0])
        actions.append(a)
        states.append(s.copy())
    return np.stack(frames), np.stack(actions), np.stack(states)


def make_transitions(n, rng=None, ep_len=24):
    """collect (pair_t, action, pair_{t+1}) where a pair is two stacked frames.
    also returns the true current state, used only for evaluation."""
    rng = rng or np.random.default_rng(0)
    p0, acts, p1, st_cur = [], [], [], []
    env = BouncingBall(rng)
    while len(p0) < n:
        frames, actions, states = _rollout_frames(env, ep_len, rng)
        # frames[i] is the frame at step i; states[i] is the state AFTER action i
        for t in range(1, len(frames) - 1):
            p0.append(frames[t - 1:t + 1])          # (prev, current)
            acts.append(actions[t])                  # action taken at current
            p1.append(frames[t:t + 2])               # (current, next)
            st_cur.append(states[t - 1])             # true state at "current"
            if len(p0) >= n:
                break
    return (
        np.stack(p0).astype(np.float32), np.stack(acts).astype(np.float32),
        np.stack(p1).astype(np.float32), np.stack(st_cur).astype(np.float32),
    )


def make_episode(k, rng=None):
    """one clean rollout: the first stacked pair, the k actions, and the k true
    states that followed. used to test multi-step prediction."""
    rng = rng or np.random.default_rng(1)
    env = BouncingBall(rng)
    frames, actions, states = _rollout_frames(env, k + 1, rng)
    first_pair = frames[0:2]                          # (frame_0, frame_1)
    return (
        first_pair.astype(np.float32),
        actions[1:k + 1].astype(np.float32),         # actions after the first pair
        states[1:k + 1],                             # matching true states
    )
