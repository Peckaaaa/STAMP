"""Generative trajectory head: where the targets go, as a distribution.

The planner scores candidate camera commands against predicted target motion, so
what it needs is a *sample* of plausible futures, not a mean.  A regression head
answers "where on average" -- and the average of a target that may turn left or
right is straight ahead, which is exactly the command a camera should not
follow.  Conditional flow matching gives the whole displacement tensor jointly,
so one sample is one coherent future for every target over the whole horizon
rather than a per-step per-target average.

Flow matching rather than a denoising chain: training is a single regression
onto a velocity, sampling is a handful of Euler steps, and the same forward pass
that gives the loss also gives an implied clean sample -- which is what the
consensus term and the reported error are computed on, with no second pass and
no backpropagation through a sampler.

    x_1        the true per-step displacements, scaled to O(1)
    x_0        ~ N(0, I)
    x_tau      (1 - tau) x_0 + tau x_1
    target     x_1 - x_0
    loss       || v_theta(x_tau, tau, b) - (x_1 - x_0) ||^2, believed slots only
    implied    x_tau + (1 - tau) v_theta        -- the model's guess at x_1

Displacements, not absolute positions: an unmoving target is the zero
prediction and the network never has to learn the map's coordinates.  They are
measured from the position the *belief* holds, which for a stale slot is where
the target was last seen, so the head also learns to correct for staleness --
the part a camera cannot get from its own eyes.

Two losses, weighted and logged apart:

``prediction``  the anchor -- true future positions out of the global state.
                Training-time information a camera never sees, so this stays
                inside CTDE.
``consensus``   two cameras that both believe in a target should predict the
                same path for it.  Alone it has a trivial optimum (predict a
                constant everywhere), so it carries a small weight beside the
                anchor and the two are reported separately: agreement that rises
                while accuracy falls is the failure to watch for.
"""

import math

import torch
import torch.nn as nn


def mlp(sizes, activation=nn.SiLU):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(activation())
    return nn.Sequential(*layers)


def time_embedding(tau, dim):
    """Sinusoidal embedding of the flow time ``tau`` in ``[0, 1]``.  ``(B,)`` -> ``(B, dim)``."""

    half = dim // 2
    frequencies = torch.exp(
        -math.log(10_000.0)
        * torch.arange(half, device=tau.device, dtype=torch.float32)
        / half
    )
    args = tau.float().unsqueeze(-1) * frequencies.unsqueeze(0) * 1000.0
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class FlowTrajectoryHead(nn.Module):
    """``b^i -> p(displacements of every believed target over H steps)``."""

    def __init__(
        self,
        belief_dim,
        n_targets,
        horizon,
        hidden_dim=256,
        displacement_scale=75.0,
        sample_steps=4,
    ):
        super().__init__()

        assert belief_dim % n_targets == 0, (belief_dim, n_targets)
        self.n_targets = n_targets
        self.slot_dim = belief_dim // n_targets
        self.horizon = horizon
        self.sample_dim = n_targets * horizon * 2
        self.sample_steps = sample_steps
        # Targets move about 0.013 belief units per step per axis in this
        # scenario, so unscaled labels would sit two orders of magnitude inside a
        # unit Gaussian prior and the velocity field would spend its capacity on
        # the scale instead of the shape.
        self.displacement_scale = displacement_scale

        self.context = mlp([belief_dim, hidden_dim, hidden_dim])
        self.time = mlp([hidden_dim, hidden_dim, hidden_dim])
        self.velocity_net = mlp(
            [self.sample_dim + 2 * hidden_dim, hidden_dim, hidden_dim, self.sample_dim]
        )
        self.hidden_dim = hidden_dim

    # ----------------------------------------------------------------- geometry

    def slots(self, belief):
        return belief.view(belief.shape[0], self.n_targets, self.slot_dim)

    def believed(self, belief):
        """``(B, n_targets)`` mask: slots this camera has any knowledge of."""

        # The wrapper keeps the known bit last in every slot for exactly this.
        return (self.slots(belief)[..., -1] > 0.5).float()

    def base_positions(self, belief):
        """``(B, n_targets, 2)``: where the belief currently puts each target."""

        return self.slots(belief)[..., :2]

    def positions(self, belief, displacement):
        """Accumulate scaled displacements from the believed position.

        ``displacement`` is ``(B, n_targets, horizon, 2)`` in model units; the
        result is in belief units, which is what the labels use.
        """

        walked = displacement.cumsum(dim=2) / self.displacement_scale
        return self.base_positions(belief).unsqueeze(2) + walked

    def displacement_targets(self, belief, future_positions):
        """True per-step displacements in model units.

        The first step is measured from the *believed* position, so a stale slot
        carries the correction the camera would otherwise never learn.
        """

        previous = torch.cat(
            [self.base_positions(belief).unsqueeze(2), future_positions[:, :, :-1]], dim=2
        )
        return (future_positions - previous) * self.displacement_scale

    # ------------------------------------------------------------------ velocity

    def velocity(self, noisy, tau, belief):
        """``v_theta(x_tau, tau, b)``; ``noisy`` is ``(B, n_targets, horizon, 2)``."""

        batch = noisy.shape[0]
        conditioning = torch.cat(
            [self.context(belief), self.time(time_embedding(tau, self.hidden_dim))], dim=-1
        )
        flat = noisy.reshape(batch, self.sample_dim)
        return self.velocity_net(torch.cat([flat, conditioning], dim=-1)).view_as(noisy)

    def implied_clean(self, noisy, tau, velocity):
        """The model's guess at ``x_1`` from one velocity evaluation."""

        return noisy + (1.0 - tau).view(-1, 1, 1, 1) * velocity

    # ------------------------------------------------------------------ sampling

    @torch.no_grad()
    def sample(self, belief, steps=None, generator=None):
        """Euler-integrate the flow from noise.  ``(B, n_targets, horizon, 2)``."""

        steps = steps or self.sample_steps
        shape = (belief.shape[0], self.n_targets, self.horizon, 2)
        x = torch.randn(shape, device=belief.device, generator=generator)

        dt = 1.0 / steps
        for index in range(steps):
            tau = torch.full((belief.shape[0],), index * dt, device=belief.device)
            x = x + dt * self.velocity(x, tau, belief)
        return x

    @torch.no_grad()
    def predict_positions(self, belief, steps=None, generator=None):
        """One sampled future per camera, in belief units, with its mask."""

        displacement = self.sample(belief, steps, generator)
        return self.positions(belief, displacement), self.believed(belief)


# ----------------------------------------------------------------------- losses


def masked_mse(predicted, target, mask):
    """Mean squared error over believed slots only.

    ``predicted``/``target`` are ``(B, n_targets, horizon, 2)`` and ``mask`` is
    ``(B, n_targets)``.  Slots the camera knows nothing about are excluded:
    asking it to guess a target it has never heard of would train it to
    hallucinate.
    """

    error = ((predicted - target) ** 2).mean(dim=(2, 3))
    return (error * mask).sum() / mask.sum().clamp(min=1.0)


def consensus_loss(positions, mask):
    """Disagreement between cameras that both believe in the same target.

    ``positions`` is ``(B, cameras, n_targets, horizon, 2)`` and ``mask`` is
    ``(B, cameras, n_targets)``.  Returns the loss and how many ordered pairs it
    was averaged over -- zero pairs means the term is inert, which is what
    happens when the communication graph is complete and every camera holds the
    same merged belief.

    Every ordered pair of cameras that both believe in a slot counts, not only
    the pairs joined by a peer-to-peer link.  Two cameras out of range can still
    both believe in a target -- through a shared neighbour, or by seeing it
    themselves -- and when they do, their predictions still have to agree, since
    the belief is supposed to mean the same thing everywhere.  Restricting the
    term to linked pairs would also need the adjacency matrix stored beside
    every window in the buffer, and the loss is small enough (measured 4e-4, at
    weight 0.1) that nothing downstream could tell the two apart.
    """

    difference = positions.unsqueeze(1) - positions.unsqueeze(2)
    error = (difference ** 2).mean(dim=(-1, -2))                 # (B, C, C, n_targets)

    pair_mask = mask.unsqueeze(1) * mask.unsqueeze(2)
    cameras = mask.shape[1]
    off_diagonal = 1.0 - torch.eye(cameras, device=mask.device).view(1, cameras, cameras, 1)
    pair_mask = pair_mask * off_diagonal

    return (error * pair_mask).sum() / pair_mask.sum().clamp(min=1.0), pair_mask.sum()


class TrajectoryFlowLearner:
    """The head, its optimizer, and the two losses that shape it.

    The only learned component in the unified pipeline: the planner replans from
    scratch every decision, so nothing else carries weights.
    """

    def __init__(
        self,
        belief_dim,
        n_targets,
        horizon,
        hidden_dim=256,
        displacement_scale=75.0,
        sample_steps=4,
        lr=3e-4,
        consensus_weight=0.1,
        max_grad_norm=1.0,
        terrain_size=1000.0,
        device='cpu',
    ):
        self.device = torch.device(device)
        self.head = FlowTrajectoryHead(
            belief_dim=belief_dim,
            n_targets=n_targets,
            horizon=horizon,
            hidden_dim=hidden_dim,
            displacement_scale=displacement_scale,
            sample_steps=sample_steps,
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.head.parameters(), lr=lr)
        self.consensus_weight = consensus_weight
        self.max_grad_norm = max_grad_norm
        self.terrain_size = terrain_size
        self.horizon = horizon

    def update(self, beliefs, future_positions):
        """One flow-matching step on ``(B, cameras, belief_dim)`` and its labels.

        ``future_positions`` is ``(B, n_targets, horizon, 2)`` in belief units --
        target-major, the layout the head predicts in.
        """

        batch, cameras = beliefs.shape[0], beliefs.shape[1]
        flat = beliefs.reshape(batch * cameras, -1)

        labels = (
            future_positions.unsqueeze(1)
            .expand(batch, cameras, *future_positions.shape[1:])
            .reshape(batch * cameras, *future_positions.shape[1:])
        )

        clean = self.head.displacement_targets(flat, labels)
        noise = torch.randn_like(clean)
        tau = torch.rand(clean.shape[0], device=clean.device)
        noisy = (1.0 - tau).view(-1, 1, 1, 1) * noise + tau.view(-1, 1, 1, 1) * clean

        velocity = self.head.velocity(noisy, tau, flat)
        mask = self.head.believed(flat)
        prediction = masked_mse(velocity, clean - noise, mask)

        # The same forward pass read as a guess at the clean sample.
        implied = self.head.implied_clean(noisy, tau, velocity)
        positions = self.head.positions(flat, implied)

        agreement, pairs = consensus_loss(
            positions.view(batch, cameras, *positions.shape[1:]),
            mask.view(batch, cameras, -1),
        )
        loss = prediction + self.consensus_weight * agreement

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.head.parameters(), self.max_grad_norm)
        self.optimizer.step()

        with torch.no_grad():
            error = ((positions - labels) ** 2).mean(dim=(2, 3))
            rmse = ((error * mask).sum() / mask.sum().clamp(min=1.0)).sqrt()

        return {
            'traj/prediction_loss': prediction.item(),
            'traj/consensus_loss': agreement.item(),
            'traj/rmse_units': rmse.item() * self.terrain_size,
            'traj/consensus_pairs': pairs.item() / max(batch, 1),
            'belief/believed_fraction': mask.mean().item(),
        }

    @torch.no_grad()
    def predict(self, beliefs):
        """``(cameras, belief_dim)`` tensor -> sampled positions in map units, and the mask."""

        positions, mask = self.head.predict_positions(beliefs)
        return positions * self.terrain_size, mask

    def state_dict(self):
        return {'head': self.head.state_dict(), 'optimizer': self.optimizer.state_dict()}

    def load_state_dict(self, d):
        self.head.load_state_dict(d['head'])
        self.optimizer.load_state_dict(d['optimizer'])
