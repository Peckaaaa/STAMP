"""Online planning over sampled target trajectories -- no actor, no policy gradient.

At every decision the whole team samples ``K`` candidate command sequences,
rolls its optics forward under them, scores how much of the *generated* target
motion each sequence would cover, and executes the softmax-weighted average.
Nothing here is trained; the trajectory head is the only learned component in
the loop, and the moment its samples are good the cameras point where the
targets are going instead of where they were.

Why a geometric rollout rather than a learned dynamics model for the *camera*: a
camera's dynamics are exactly known and closed-form -- MATE rotates by the
commanded angle, clips the viewing angle, and rescales the sight range to keep
the sector area constant -- and detection is a distance test and an angle test.
Rolling those forward is arithmetic.  The generative model is spent where the
uncertainty actually is, which is the targets.

Everything is torch and batched over cameras and candidates together, so a
decision is a handful of kernel launches whatever the team size:

    candidates   (n_agents, K, H, 2)
    optics       (n_agents, K, H)      orientation, viewing angle, sight range
    coverage     (n_agents, K, H, n_targets)
    rewards      (n_agents, K)

Two approximations, both deliberate:

  Obstacles are ignored while scoring.  MATE lets a camera see a target behind
  an obstacle with the obstacle's transmittance, so occlusion changes the score
  by a probability rather than a hard zero, and the planner would need obstacle
  geometry it does not carry.

  Detection is scored softly.  A hard count is flat almost everywhere -- most
  sampled sequences cover the same integer number of targets -- and a softmax
  over a flat landscape is a coin flip.  Soft margins keep "nearly in view"
  distinguishable from "hopeless".

Coordination is decentralized and one step stale: each camera publishes what its
plan intends to cover on the same peer-to-peer channel that carries the belief,
and discounts targets its neighbours claimed on the previous decision.  Without
it, cameras that share a belief plan the same sweep, since coverage counts
distinct targets and nothing in an independent score says so.
"""

import math

import numpy as np
import torch


def wrap_degrees(angles):
    """Signed angular difference in degrees, wrapped into ``[-180, 180)``."""

    return (angles + 180.0) % 360.0 - 180.0


class CameraKinematics:
    """MATE's camera dynamics for the whole team, read out of their private states.

    A private state is ``[x, y, radius, Rs cos(phi), Rs sin(phi), theta, Rs_max,
    rotation_step, zooming_step]``: the sight vector is stored in cartesian form,
    so range and orientation come back out of it.  The conserved sector area
    ``theta * Rs^2`` gives the minimum viewing angle, which the clip needs and
    which is not otherwise in the state.
    """

    MAX_VIEWING_ANGLE = 180.0

    def __init__(self, camera_states):
        state = torch.as_tensor(camera_states, dtype=torch.float32)
        # Every derived quantity below inherits this tensor's device.

        self.location = state[:, 0:2]                                    # (n, 2)
        sight_vector = state[:, 3:5]
        self.sight_range = sight_vector.norm(dim=-1)                     # (n,)
        self.orientation = torch.rad2deg(
            torch.atan2(sight_vector[:, 1], sight_vector[:, 0])
        )
        self.viewing_angle = state[:, 5]
        self.max_sight_range = state[:, 6]

        self.area_product = self.viewing_angle * self.sight_range ** 2
        self.min_viewing_angle = self.area_product / self.max_sight_range.clamp(min=1e-8) ** 2

    def roll(self, actions):
        """``(n, K, H, 2)`` commands in MATE units -> optics at every step.

        Each returned tensor is ``(n, K, H)``.  The clips are the environment's
        own, not a second policy.
        """

        orientation = self.orientation.view(-1, 1, 1) + actions[..., 0].cumsum(dim=-1)
        # A tensor lower bound and a scalar upper bound cannot be mixed in one
        # clamp call, and the lower bound is per camera.
        viewing_angle = torch.minimum(
            torch.maximum(
                self.viewing_angle.view(-1, 1, 1) + actions[..., 1].cumsum(dim=-1),
                self.min_viewing_angle.view(-1, 1, 1),
            ),
            torch.full_like(self.viewing_angle.view(-1, 1, 1), self.MAX_VIEWING_ANGLE),
        )
        sight_range = torch.sqrt(self.area_product.view(-1, 1, 1) / viewing_angle)
        return wrap_degrees(orientation), viewing_angle, sight_range


class MPPIPlanner:
    """Model-predictive path integral control over the whole camera team."""

    def __init__(
        self,
        horizon=4,
        samples=64,
        temperature=0.05,
        noise_scale=0.6,
        discount=0.95,
        range_softness=60.0,
        angle_softness=6.0,
        intent_discount=0.8,
        recall_weight=0.0,
        recall_tau=5.0,
        recall_max_age=8.0,
        recall_hypotheses=4,
        recall_drift=17.0,
        explore_weight=0.0,
        explore_voronoi=True,
        angle_weight=0.0,
        angle_threshold=2.0,
        seed=0,
        device='cpu',
    ):
        self.horizon = horizon
        self.samples = samples
        self.temperature = temperature
        self.noise_scale = noise_scale
        self.discount = discount
        self.range_softness = range_softness
        self.angle_softness = angle_softness
        self.intent_discount = intent_discount
        # Active search.  All three terms live in the reward rather than in any
        # weights, so a configuration can be scored on a checkpoint trained
        # without them, and a zero weight is exactly the policy before it.
        self.recall_weight = recall_weight
        self.recall_tau = recall_tau
        self.recall_max_age = recall_max_age
        self.recall_hypotheses = int(recall_hypotheses)
        self.recall_drift = recall_drift
        self.explore_weight = explore_weight
        self.explore_voronoi = bool(explore_voronoi)
        self.angle_weight = angle_weight
        self.angle_threshold = angle_threshold
        self.device = torch.device(device)
        self.generator = torch.Generator(device=self.device).manual_seed(int(seed))
        self.mean = None

    def reset(self, n_agents, action_dim):
        """Warm starts, one plan per camera, cleared at an episode boundary."""

        self.mean = torch.zeros(n_agents, self.horizon, action_dim, device=self.device)

    # ------------------------------------------------------------------ scoring

    def _coverage(self, kinematics, orientation, viewing_angle, sight_range, positions):
        """``(n, K, H, n_targets)`` soft detection scores.

        ``positions`` is ``(n, n_targets, H, 2)`` in map units: where the
        trajectory head sampled each target to be at each step of the plan.
        """

        relative = positions - kinematics.location.view(-1, 1, 1, 2)     # (n, T, H, 2)
        distance = relative.norm(dim=-1).transpose(1, 2).unsqueeze(1)    # (n, 1, H, T)
        bearing = torch.rad2deg(
            torch.atan2(relative[..., 1], relative[..., 0])
        ).transpose(1, 2).unsqueeze(1)                                   # (n, 1, H, T)

        in_range = torch.sigmoid(
            (sight_range.unsqueeze(-1) - distance) / self.range_softness
        )
        offset = wrap_degrees(bearing - orientation.unsqueeze(-1)).abs()
        in_view = torch.sigmoid(
            (0.5 * viewing_angle.unsqueeze(-1) - offset) / self.angle_softness
        )
        return in_range * in_view

    def _discounted(self, per_step):
        """``(n, K, H)`` -> ``(n, K)``, later steps worth less."""

        discount = self.discount ** torch.arange(
            per_step.shape[-1], device=per_step.device, dtype=per_step.dtype
        )
        return (per_step * discount.view(1, 1, -1)).sum(dim=-1)

    def _score(self, coverage, weight):
        """Discounted soft coverage of anything, weighted per item.

        ``coverage`` is ``(n, K, H, M)`` and ``weight`` either ``(n, M)`` -- one
        worth per item -- or ``(n, 1, H, M)`` when the worth moves with the roll.
        """

        if weight.dim() == 2:
            weight = weight.view(weight.shape[0], 1, 1, -1)
        return self._discounted((coverage * weight).sum(dim=-1))

    def _rewards(self, coverage, believed, peer_intent):
        """Discounted soft coverage of believed targets, minus what peers claimed."""

        weight = believed * (1.0 - self.intent_discount * peer_intent)   # (n, T)
        return self._score(coverage, weight)

    # ------------------------------------------------------------- active search

    def _explore_term(self, kinematics, optics, search):
        """Reward for sweeping map cells nobody has looked at for a while.

        This is the term that answers the hole in the tracking reward: a target
        no camera believes in has weight exactly zero there, so nothing pays for
        going to look for it.  A cell's worth is how long it has gone unseen, so
        the cameras are paid to cover *area* where the unseen targets must be,
        without anyone having to guess which slot is where.

        Voronoi: without it every camera turns towards the same stalest corner,
        because they all read (nearly) the same map.  Assigning each cell to its
        nearest camera splits the map with no extra byte on the wire -- camera
        positions are fixed and already known to everyone.
        """

        centres = search['cell_centres']                                 # (C, 2)
        weight = search['staleness']                                     # (n, C)
        if self.explore_voronoi:
            distance = torch.cdist(kinematics.location, centres)         # (n, C)
            owner = distance.argmin(dim=0)
            mine = torch.nn.functional.one_hot(
                owner, num_classes=kinematics.location.shape[0]
            ).transpose(0, 1).to(weight.dtype)                           # (n, C)
            weight = weight * mine

        n_agents, horizon = kinematics.location.shape[0], optics[0].shape[-1]
        positions = centres.view(1, -1, 1, 2).expand(n_agents, -1, horizon, 2)
        coverage = self._coverage(kinematics, *optics, positions)
        # Summed over the horizon like the tracking term, not averaged.  An
        # earlier version divided by the horizon to keep the two terms on the
        # same scale; measured, that made `explore_weight` four times weaker
        # than the value it is documented at, and the optimum moved to 0.4.
        # Same reward, different parametrization -- this is the one the
        # measured plateau at 0.075-0.15 belongs to.
        return self._score(coverage, weight)

    def _recall_term(self, kinematics, optics, search):
        """Reward for revisiting where a lost target could have drifted to.

        A slot that went unseen `age` decisions ago is somewhere on a ring of
        radius ``drift * age`` around its last fix -- measured at 17 units per
        decision in this scenario.  The ring is sampled at a few points rather
        than integrated: the planner only needs to tell "this sweep passes
        through the plausible set" from "this one does not".

        Confidence decays with age and the term switches off past
        ``recall_max_age``, where the ring is wider than a sector and paying for
        it is paying for nothing.
        """

        last_seen = search['last_seen']                                  # (n, T, 2)
        age = search['age']                                              # (n, T)
        known = search['known']
        n_agents, n_targets = age.shape
        horizon = optics[0].shape[-1]
        hypotheses = max(self.recall_hypotheses, 1)

        angles = torch.arange(hypotheses, device=age.device, dtype=age.dtype)
        angles = 2.0 * math.pi * angles / hypotheses
        ring = torch.stack([angles.cos(), angles.sin()], dim=-1)         # (m, 2)

        # The ring keeps growing over the roll: by step h the target has had
        # `age + h + 1` decisions to drift.
        steps = torch.arange(1, horizon + 1, device=age.device, dtype=age.dtype)
        radius = self.recall_drift * (age.unsqueeze(-1) + steps.view(1, 1, -1))
        offsets = radius.unsqueeze(-1).unsqueeze(-1) * ring.view(1, 1, 1, -1, 2)
        positions = last_seen.view(n_agents, n_targets, 1, 1, 2) + offsets
        positions = positions.permute(0, 1, 3, 2, 4).reshape(
            n_agents, n_targets * hypotheses, horizon, 2
        )

        eligible = (1.0 - known) * (age > 0.0) * (age <= self.recall_max_age)
        confidence = torch.exp(-age / max(self.recall_tau, 1e-6)) * eligible
        weight = (confidence / hypotheses).unsqueeze(-1).expand(-1, -1, hypotheses)
        weight = weight.reshape(n_agents, n_targets * hypotheses)

        coverage = self._coverage(kinematics, *optics, positions)
        return self._score(coverage, weight)

    def _angle_term(self, kinematics, viewing_angle, tracked):
        """Reward for opening the aperture when there is little left to track.

        The optics trade width for reach at constant area, so a camera with
        nothing believed in front of it is better off wide: a wide sector is more
        likely to catch whatever wanders in.  The bonus fades out as soon as the
        camera has ``angle_threshold`` targets' worth to follow, so it never
        competes with tracking.
        """

        deficit = (self.angle_threshold - tracked.sum(dim=-1)).clamp(min=0.0)
        deficit = (deficit / max(self.angle_threshold, 1e-6)).view(-1, 1)
        width = viewing_angle / CameraKinematics.MAX_VIEWING_ANGLE
        return deficit * self._discounted(width)

    def _search_tensors(self, search):
        """Move the environment's spatial memory onto the device, once.

        Returns ``None`` when there is nothing to move or no term would read it,
        which is what keeps the tracking-only configuration free of the cost.
        """

        if search is None:
            return None
        if not (self.explore_weight or self.recall_weight):
            return None
        return {
            key: torch.as_tensor(value, dtype=torch.float32, device=self.device)
            for key, value in search.items()
        }

    # --------------------------------------------------------------------- plan

    @torch.no_grad()
    def plan(self, camera_states, predicted_positions, believed, peer_intent,
             action_low, action_high, search=None):
        """One decision for the whole team.

        Args:
            camera_states: ``(n, 9)`` raw private camera states.
            predicted_positions: ``(n, n_targets, horizon, 2)`` in map units.
            believed: ``(n, n_targets)`` which slots each camera knows.
            peer_intent: ``(n, n_targets)`` what neighbours claimed last decision.
            action_low, action_high: MATE's camera action box.
            search: the environment's spatial memory -- cell centres, per-camera
                staleness, and where each slot was last fixed -- in map units.
                ``None``, or every search weight left at zero, is the tracking
                policy exactly.

        Returns:
            ``(actions in [-1, 1], intent)`` as tensors -- the command to execute
            and what it means to cover, for the next round of the channel.
        """

        tensor = lambda x: torch.as_tensor(x, dtype=torch.float32, device=self.device)
        camera_states = tensor(camera_states)
        predicted_positions = tensor(predicted_positions)
        believed = tensor(believed)
        peer_intent = tensor(peer_intent)
        low, high = tensor(action_low), tensor(action_high)

        n_agents, action_dim = camera_states.shape[0], low.shape[0]
        if self.mean is None or self.mean.shape[0] != n_agents:
            self.reset(n_agents, action_dim)

        noise = torch.randn(
            (n_agents, self.samples, self.horizon, action_dim),
            device=self.device,
            generator=self.generator,
        ) * self.noise_scale * (high - low) / 2.0
        candidates = torch.clamp(self.mean.unsqueeze(1) + noise, low, high)
        # One candidate is the warm start itself, so a good plan is never lost to
        # sampling.
        candidates[:, 0] = torch.clamp(self.mean, low, high)

        # `camera_states` is already on the device, so the kinematics are built
        # there: no host round trip inside a decision.
        kinematics = CameraKinematics(camera_states)

        optics = kinematics.roll(candidates)
        coverage = self._coverage(kinematics, *optics, predicted_positions)
        orientation, viewing_angle, sight_range = optics
        rewards = self._rewards(coverage, believed, peer_intent)         # (n, K)

        search = self._search_tensors(search)
        if search is not None:
            if self.explore_weight:
                rewards = rewards + self.explore_weight * self._explore_term(
                    kinematics, optics, search
                )
            if self.recall_weight:
                rewards = rewards + self.recall_weight * self._recall_term(
                    kinematics, optics, search
                )
        if self.angle_weight:
            rewards = rewards + self.angle_weight * self._angle_term(
                kinematics, viewing_angle, believed * (1.0 - peer_intent)
            )

        weights = torch.softmax(rewards / max(self.temperature, 1e-8), dim=-1)
        plan = (weights.view(n_agents, -1, 1, 1) * candidates).sum(dim=1)  # (n, H, 2)

        actions = plan[:, 0]
        # Shift the warm start: what was planned for the next step becomes this
        # camera's starting guess at the next decision.
        self.mean = torch.cat(
            [plan[:, 1:], torch.zeros(n_agents, 1, action_dim, device=self.device)], dim=1
        )

        # What the executed plan expects to cover, for the neighbours.
        expected = (weights.view(n_agents, -1, 1, 1) * coverage).sum(dim=1)  # (n, H, T)
        intent = (expected.max(dim=1).values * believed).clamp(0.0, 1.0)

        normalized = 2.0 * (actions - low) / (high - low) - 1.0
        return normalized, intent

    @staticmethod
    def to_numpy(tensor):
        return tensor.detach().cpu().numpy().astype(np.float32)
